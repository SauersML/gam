//! Builds the unified ψ `HyperCoord` objects + pair/drift callbacks from the
//! family-provided penalty derivatives, and evaluates the custom-family
//! hyper-objective (joint and EFS variants, shared + public entry points).

use super::*;

/// Materialize `D_beta(∂_psi H_info)[direction]` from the same ψ authority
/// that supplied `∂_psi H_info`.
///
/// An exact-ψ workspace owns more than a representation cache: it also owns
/// the evaluation options and row measure used by its first-order terms. In
/// particular, survival marginal-slope's early outer pilot installs a
/// Horvitz-Thompson row measure in that workspace. Replaying the direct family
/// hook here would combine a workspace `∂_psi H_info` with a full-data
/// `D_beta(∂_psi H_info)` inside one Jeffreys derivative. That pair is not the
/// derivative of any information matrix.
///
/// This is the materializing counterpart of
/// [`build_psi_drift_deriv_callback`]: a present workspace is the exclusive
/// authority, while the direct family hook is used only when no workspace was
/// constructed.
fn materialize_authoritative_psi_hessian_directional_derivative<
    F: CustomFamily + Clone + Send + Sync + 'static,
>(
    family: &F,
    synced_states: &[ParameterBlockState],
    specs: &[ParameterBlockSpec],
    hyper_layout: &CustomFamilyHyperLayout,
    psi_workspace: Option<&dyn ExactNewtonJointPsiWorkspace>,
    psi_index: usize,
    direction: &Array1<f64>,
    total: usize,
) -> Result<Option<Array2<f64>>, CustomFamilyError> {
    let drift = if let Some(workspace) = psi_workspace {
        workspace.hessian_directional_derivative(psi_index, direction)?
    } else {
        family
            .exact_newton_joint_psihessian_directional_derivative(
                synced_states,
                specs,
                hyper_layout,
                psi_index,
                direction,
            )?
            .map(DriftDerivResult::Dense)
    };
    match drift {
        Some(DriftDerivResult::Dense(matrix)) => {
            if matrix.dim() != (total, total) {
                return Err(CustomFamilyError::trial_point(format!(
                    "authoritative psi Hessian directional derivative for axis {psi_index} \
                     has dense shape {:?}, expected ({total}, {total})",
                    matrix.dim(),
                )));
            }
            Ok(Some(matrix))
        }
        Some(DriftDerivResult::Operator(operator)) => {
            if operator.dim() != total {
                return Err(CustomFamilyError::trial_point(format!(
                    "authoritative psi Hessian directional derivative for axis {psi_index} \
                     has operator dimension {}, expected {total}",
                    operator.dim(),
                )));
            }
            Ok(Some(operator.mul_mat(&Array2::<f64>::eye(total))))
        }
        None => Ok(None),
    }
}

/// [`materialize_authoritative_psi_hessian_directional_derivative`] along every
/// joint coefficient axis at one ψ axis, in axis order, from the workspace when
/// one is present and from the family otherwise — each of which may build the
/// whole set from one row sweep (gam#979). `None` means no axis is available,
/// exactly as the per-axis materializer's `None` on the first axis would.
fn materialize_authoritative_psi_hessian_directional_derivatives_all_beta_axes<
    F: CustomFamily + Clone + Send + Sync + 'static,
>(
    family: &F,
    synced_states: &[ParameterBlockState],
    specs: &[ParameterBlockSpec],
    hyper_layout: &CustomFamilyHyperLayout,
    psi_workspace: Option<&dyn ExactNewtonJointPsiWorkspace>,
    psi_index: usize,
    total: usize,
) -> Result<Option<Vec<Array2<f64>>>, CustomFamilyError> {
    let drifts = if let Some(workspace) = psi_workspace {
        workspace.hessian_directional_derivatives_all_beta_axes(psi_index, total)?
    } else {
        family
            .exact_newton_joint_psihessian_directional_derivatives_all_beta_axes(
                synced_states,
                specs,
                hyper_layout,
                psi_index,
            )?
            .map(|axes| axes.into_iter().map(DriftDerivResult::Dense).collect::<Vec<_>>())
    };
    let Some(drifts) = drifts else {
        return Ok(None);
    };
    if drifts.len() != total {
        return Err(CustomFamilyError::trial_point(format!(
            "authoritative psi Hessian all-axes derivative for axis {psi_index} produced {} \
             axes, expected {total}",
            drifts.len()
        )));
    }
    let mut axes = Vec::with_capacity(total);
    for drift in drifts {
        axes.push(match drift {
            DriftDerivResult::Dense(matrix) => {
                if matrix.dim() != (total, total) {
                    return Err(CustomFamilyError::trial_point(format!(
                        "authoritative psi Hessian all-axes derivative for axis {psi_index} \
                         has dense shape {:?}, expected ({total}, {total})",
                        matrix.dim(),
                    )));
                }
                matrix
            }
            DriftDerivResult::Operator(operator) => {
                if operator.dim() != total {
                    return Err(CustomFamilyError::trial_point(format!(
                        "authoritative psi Hessian all-axes derivative for axis {psi_index} \
                         has operator dimension {}, expected {total}",
                        operator.dim(),
                    )));
                }
                operator.mul_mat(&Array2::<f64>::eye(total))
            }
        });
    }
    Ok(Some(axes))
}

/// Build `HyperCoord` objects for ψ (custom family) hyperparameters.
///
/// Converts family-provided (a^ℓ, q, L) objects and penalty derivatives
/// into the unified (a, g, B, ld_s) format. Each ψ coordinate produces
/// one `HyperCoord` in the flattened joint coefficient space.
///
/// The mapping from family objects to HyperCoord is:
///
///   a    = a^ℓ_ψ + 0.5 β̂^T S_ψ β̂
///   g    = q_ψ + S_ψ β̂
///   B    = L_ψ + S_ψ
///   ld_s = tr(S₊⁻¹ S_ψ)
///
/// where S_ψ is the assembled penalty derivative in joint coefficient space.
pub fn build_psi_hyper_coords<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    synced_states: &[ParameterBlockState],
    specs: &[ParameterBlockSpec],
    hyper_layout: &CustomFamilyHyperLayout,
    beta_flat: &Array1<f64>,
    rho: &[f64],
    penalty_counts: &[usize],
    s_logdet_blocks: Option<&[PenaltyPseudologdet]>,
    hessian_beta_independent: bool,
    psi_workspace: Option<Arc<dyn ExactNewtonJointPsiWorkspace>>,
) -> Result<Vec<HyperCoord>, CustomFamilyError> {
    let ranges = block_param_ranges(specs);
    let total = beta_flat.len();
    let per_block = split_log_lambdas(&Array1::from_vec(rho.to_vec()), penalty_counts)?;
    let per_block_lambdas =
        exact_lambdas_by_block(&per_block, "psi hyper log strength")?;

    let mut coords = Vec::new();

    let build_psi_hyper_coords_start = std::time::Instant::now();
    let total_axes = hyper_layout.len();

    let batched_terms: Option<Vec<ExactNewtonJointPsiTerms>> = match psi_workspace.as_ref() {
        Some(workspace) => workspace.first_order_terms_all()?,
        None => None,
    };
    if let Some(terms) = batched_terms.as_ref()
        && terms.len() != total_axes
    {
        return Err(CustomFamilyError::trial_point(format!(
            "custom-family hyper workspace returned {} first-order axes for layout length {total_axes}",
            terms.len()
        )));
    }

    // EXPLICIT ∂_ρ H_Φ context (gam#854). The joint-Jeffreys curvature `H_Φ` is
    // built from the JOINT Hessian `H_joint(β, ρ)`, so for a family whose
    // `H_joint` depends on a ψ hyperparameter (the adaptive penalty's `λ_m`/`ε_m`,
    // or any penalty folded into `H_joint`) it depends on ρ EXPLICITLY, not only
    // through β̂. The augmented-LAML score `½ tr[(H+S_λ+H_Φ)⁻¹ ∂_ρ(H+S_λ+H_Φ)]` then
    // needs the explicit term `∂_ρ_i H_Φ|_β` added to each ψ coord's drift (the
    // mode-response part `D_β H_Φ[v_k]` is already folded in elsewhere). We form it
    // from the SAME pieces the value path uses — the full identifiable Jeffreys span
    // `Z_J` and the snapshot joint Hessian `H_joint(β̂)` — once per evaluation, and
    // contract it per coord with `∂_ρ_i H_joint|_β` (the coord drift `dense_b`) and
    // `∂_ρ_i Hdot[e_a]|_β` (the family's ψ-Hessian directional derivative). `None`
    // unless the family uses the Jeffreys term and exposes a dense joint Hessian, so
    // every non-Jeffreys / operator-only family is byte-unchanged.
    let jeffreys_hphi_ctx: Option<(Array2<f64>, Array2<f64>)> =
        if family.joint_jeffreys_term_required() && !hyper_layout.is_empty() {
            match (
                build_joint_jeffreys_subspace(family, specs, &ranges)?,
                family.joint_jeffreys_information_with_specs(synced_states, specs)?,
            ) {
                (Some(z), Some(h))
                    if z.nrows() == total && h.nrows() == total && h.ncols() == total =>
                {
                    Some((z, h))
                }
                _ => None,
            }
        } else {
            None
        };

    // Whether the Jeffreys information `H_info` depends EXPLICITLY on ψ
    // (gam#1607). When `false` (penalty/prior ψ that leave the design — hence
    // the likelihood Fisher information — fixed, e.g. spatial-adaptive
    // Charbonnier), `∂_ψ H_info|_β ≡ 0`, so the three explicit-ψ Firth terms
    // (`−∂_ψΦ`, `−∂_β∂_ψΦ`, `∂_ψ H_Φ`) vanish identically and must NOT be formed
    // from `hessian_psi` (which is `∂_ψ(penalty)`, the WRONG perturbation —
    // the penalty's ψ-derivative, not the information's). The implicit
    // β-mode-response of `Φ` (the operator `H_Φ` and its `D_β H_Φ[β̇]` drift)
    // is independent of this flag and stays folded.
    let jeffreys_info_depends_on_psi = family.joint_jeffreys_information_depends_on_psi();

    // The reduced Jeffreys spectrum — `Z_Jᵀ H_info Z_J`, its eigendecomposition,
    // the conditioning gate, the relative floor and the dominant/worst
    // eigenvalue indices — is a property of the SNAPSHOT `(H_info, Z_J)` alone.
    // It does not move with the ψ axis being differentiated, nor with the
    // coefficient axis of a mixed derivative. gam#979: every explicit-ψ Jeffreys
    // derivative below used to rebuild it from scratch, so one outer gradient
    // paid `axes × coefficients` eigendecompositions of the reduced information
    // for one distinct spectrum. Prepare it once here and hand the prepared plan
    // to each derivative; the arithmetic each of them performs is unchanged.
    let jeffreys_plan: Option<gam_solve::estimate::reml::jeffreys_subspace::JointJeffreysPlan> =
        match jeffreys_hphi_ctx
            .as_ref()
            .filter(|_| jeffreys_info_depends_on_psi)
        {
            Some((z_j, h_joint)) => Some(
                gam_solve::estimate::reml::jeffreys_subspace::JointJeffreysPlan::prepare(
                    h_joint.view(),
                    z_j.view(),
                )?,
            ),
            None => None,
        };

    // The explicit-ψ Jeffreys score and curvature use the SAME canonical
    // coefficient-axis derivatives {Hdot[e_a]}. Previously each ψ axis rebuilt
    // those p matrices once in the mixed-score loop and AGAIN while preparing
    // ∂ψH_Φ: 2·K·p full row streams at one fixed β. Acquire the family's
    // authoritative build-once all-axes batch exactly once and prepare the
    // reduced Jeffreys base from that same batch. The original per-axis
    // capability path remains below only for families without an all-axes
    // provider; there is no approximate reuse.
    let jeffreys_base_axis_derivatives: Option<Vec<Array2<f64>>> =
        if jeffreys_hphi_ctx.is_some() && jeffreys_info_depends_on_psi {
            family.joint_jeffreys_information_directional_derivative_all_axes_with_specs(
                synced_states,
                specs,
            )?
        } else {
            None
        };
    let jeffreys_hphi_base = match (
        jeffreys_hphi_ctx.as_ref(),
        jeffreys_base_axis_derivatives.as_ref(),
    ) {
        (Some((z_j, h_joint)), Some(axis_derivatives)) => {
            gam_solve::estimate::reml::jeffreys_subspace::JeffreysHphiDriftBase::prepare_with_axes(
                h_joint.view(),
                z_j.view(),
                axis_derivatives.clone(),
            )?
        }
        _ => None,
    };

    for psi_global in 0..total_axes {
        let axis = hyper_layout
            .axis(psi_global)
            .ok_or_else(|| CustomFamilyError::trial_point(format!("missing typed hyper axis {psi_global}")))?;
        // 1. Get family-provided likelihood objects (joint flattened space).
        let psi_terms = if let Some(batched) = batched_terms.as_ref() {
            Some(batched[psi_global].clone())
        } else if let Some(workspace) = psi_workspace.as_ref() {
            match workspace.first_order_terms(psi_global)? {
                Some(terms) => Some(terms),
                None => family.exact_newton_joint_psi_terms(
                    synced_states,
                    specs,
                    hyper_layout,
                    psi_global,
                )?,
            }
        } else {
            family.exact_newton_joint_psi_terms(synced_states, specs, hyper_layout, psi_global)?
        };
        let psi_terms = match (axis, psi_terms) {
            (_, Some(terms)) => terms,
            (CustomFamilyHyperAxis::DesignPenalty { .. }, None) => {
                ExactNewtonJointPsiTerms::zeros(total)
            }
            (CustomFamilyHyperAxis::Family { family_axis }, None) => {
                return Err(CustomFamilyError::trial_point(format!(
                    "family-owned hyper axis {family_axis} has no exact first-order V_i/g_i/H_i terms"
                )));
            }
        };

        // 2. Assemble generic penalty motion only for a typed design/penalty
        // axis. Family axes have no fabricated block owner and therefore carry
        // exactly zero S_i.
        let penalty_motion =
            hyper_layout
                .design_derivative(psi_global)
                .map(|(block_idx, _, deriv)| {
                    let (start, end) = ranges[block_idx];
                    let p_block = end - start;
                    let s_psi_local =
                        assemble_block_local_s_psi(deriv, &per_block_lambdas[block_idx], p_block);
                    (block_idx, start, end, s_psi_local)
                });

        // 3. Build HyperCoord using block-local S_ψ when present.
        let mut a = psi_terms.objective_psi;
        let mut s_psi_beta = Array1::zeros(total);
        if let Some((_, start, end, s_psi_local)) = penalty_motion.as_ref() {
            let beta_block = beta_flat.slice(ndarray::s![*start..*end]);
            let s_psi_beta_local = s_psi_local.dot(&beta_block);
            a += 0.5 * beta_block.dot(&s_psi_beta_local);
            s_psi_beta
                .slice_mut(ndarray::s![*start..*end])
                .assign(&s_psi_beta_local);
        }

        // EXPLICIT Firth VALUE ψ-derivative (gam#1607). The outer LAML cost folds
        // `−Φ(β̂)` where `Φ = ½ log|Z_Jᵀ H_info Z_J|₊` (gated), and the Jeffreys
        // information `H_info` is the data joint Hessian — so for a ψ hyperparameter
        // that reshapes the design (matern/duchon length-scale) it depends on ψ
        // EXPLICITLY, with `∂_ψ H_info|_β` the family's ψ-Hessian derivative (the
        // dense `hessian_psi`, or the materialized operator when the workspace path
        // streams it). The companion CURVATURE term `∂_ψ H_Φ` is added to the dense
        // drift below (gam#854); but the VALUE term `−∂_ψΦ` was dropped on EVERY ψ
        // axis (and entirely on the operator path), leaving the outer ψ-gradient
        // short by the full Firth value motion (dominant on the spatial axis). The
        // helper returns `0.0` when the conditioning gate skips the term, so a clean
        // / well-conditioned fit is byte-unchanged.
        // `∂_ψ H_info|_β` (the explicit ψ-derivative of the Jeffreys information),
        // materialized once and reused for BOTH the VALUE gradient term `−∂_ψΦ`
        // (here) and the Hessian β-coupling term `−∂_β∂_ψΦ` (the score below).
        let firth_pert_info: Option<Array2<f64>> =
            if jeffreys_hphi_ctx.is_some() && jeffreys_info_depends_on_psi {
                if let Some(op) = psi_terms.hessian_psi_operator.as_ref() {
                    Some(op.mul_mat(&ndarray::Array2::<f64>::eye(total)))
                } else if psi_terms.hessian_psi.nrows() == total
                    && psi_terms.hessian_psi.ncols() == total
                {
                    Some(psi_terms.hessian_psi.clone())
                } else {
                    None
                }
            } else {
                None
            };
        if let (Some(plan), Some(pert_info)) = (jeffreys_plan.as_ref(), firth_pert_info.as_ref()) {
            let phi_psi = plan.explicit_param_derivative(pert_info)?;
            a -= phi_psi;
        }
        let mut g = &psi_terms.score_psi + &s_psi_beta;

        // EXPLICIT Firth Hessian β-COUPLING (gam#1607). The outer Hessian's
        // mode-response term is `−g_ψ·β̇`, with the coord score `g_ψ = ∂_β∂_ψV|_β`.
        // The Firth value `−Φ(β̂)` contributes `−∂_β∂_ψΦ` to that score (β̂ moves
        // with ψ as the length-scale reshapes the design, so the Firth value's
        // ψ-gradient has a genuine β-response), EXACTLY mirroring the ρ-coord path
        // (`g_j -= gphi_τ`, gam#854/#979). The per-β-axis mixed second derivative
        // `∂_β_a∂_ψΦ` is the validated explicit second-derivative helper applied to
        // the perturbation pair `(∂_ψH_info, ∂_β_a H_info = Hdot[e_a])` with mixed
        // `∂_ψ∂_β_a H_info = ∂_ψHdot[e_a]` — the SAME family directional derivatives
        // the `∂_ψH_Φ` curvature term consumes. The helper returns `0.0` when the
        // conditioning gate skips the term, so a clean fit is byte-unchanged.
        // Materialize the ψ-mixed canonical-axis batch once for this ψ axis.
        // Both −∂β∂ψΦ below and ∂ψH_Φ below consume these exact matrices; the
        // former code called the row-streaming provider independently in each
        // location, doubling the genuinely ψ-dependent work as well as the base
        // work. Keep canonical axis order so every contraction is unchanged.
        // One sweep for all `total` axes where the family or workspace can build
        // it (gam#979: this loop used to materialize each axis with its own row
        // sweep, 58 s per gradient on the rigid marginal-slope arm).
        let firth_pert_axis_derivatives: Option<Vec<Array2<f64>>> =
            if jeffreys_hphi_ctx.is_some() && firth_pert_info.is_some() {
                materialize_authoritative_psi_hessian_directional_derivatives_all_beta_axes(
                    family,
                    synced_states,
                    specs,
                    hyper_layout,
                    psi_workspace.as_deref(),
                    psi_global,
                    total,
                )?
            } else {
                None
            };
        if let (Some(plan), Some(pert_info)) = (jeffreys_plan.as_ref(), firth_pert_info.as_ref()) {
            if let (Some(base_axes), Some(pert_axes)) = (
                jeffreys_base_axis_derivatives.as_ref(),
                firth_pert_axis_derivatives.as_ref(),
            ) {
                if base_axes.len() != total || pert_axes.len() != total {
                    return Err(CustomFamilyError::trial_point(format!(
                        "explicit-psi Jeffreys axis batch lengths ({}, {}) != coefficient dimension {total}",
                        base_axes.len(),
                        pert_axes.len(),
                    )));
                }
                // The ambient trace weights depend on the snapshot spectrum and on
                // `∂_ψH_info` — NOT on the coefficient axis. One preparation serves
                // every axis; each axis pays only its own Frobenius contraction
                // (gam#979: this preparation used to run once per coefficient axis).
                let weights = plan.explicit_param_mixed_trace_weights(pert_info)?;
                for (a_idx, (hdot_a, psi_hdot_a)) in
                    base_axes.iter().zip(pert_axes.iter()).enumerate()
                {
                    let phi_psi_beta_a = weights.contract(hdot_a, psi_hdot_a)?;
                    g[a_idx] -= phi_psi_beta_a;
                }
            } else {
                // Typed capability path for families without a build-once
                // canonical batch. Exact and intentionally isolated from the
                // authoritative batched path above.
                // Both sets from one sweep each where the family can (gam#979);
                // the per-axis sweep is the fallback for a family that answers
                // one axis at a time.
                let batched = match (
                    family.joint_jeffreys_information_directional_derivative_all_axes_with_specs(
                        synced_states,
                        specs,
                    )?,
                    materialize_authoritative_psi_hessian_directional_derivatives_all_beta_axes(
                        family,
                        synced_states,
                        specs,
                        hyper_layout,
                        psi_workspace.as_deref(),
                        psi_global,
                        total,
                    )?,
                ) {
                    (Some(hdots), Some(psi_hdots))
                        if hdots.len() == total && psi_hdots.len() == total =>
                    {
                        Some((hdots, psi_hdots))
                    }
                    _ => None,
                };
                if let Some((hdots, psi_hdots)) = batched {
                    let weights = plan.explicit_param_mixed_trace_weights(pert_info)?;
                    for a_idx in 0..total {
                        let phi_psi_beta_a =
                            weights.contract(&hdots[a_idx], &psi_hdots[a_idx])?;
                        g[a_idx] -= phi_psi_beta_a;
                    }
                } else {
                    // Prepared on the first axis the family actually serves, so an
                    // evaluation where every axis declines behaves exactly as before.
                    let mut weights: Option<
                        gam_solve::estimate::reml::jeffreys_subspace::JointJeffreysExplicitMixedTraceWeights,
                    > = None;
                    for a_idx in 0..total {
                        let mut e_a = Array1::<f64>::zeros(total);
                        e_a[a_idx] = 1.0;
                        let hdot_a =
                            family.joint_jeffreys_information_directional_derivative_with_specs(
                                synced_states,
                                specs,
                                &e_a,
                            )?;
                        let psi_hdot_a =
                            materialize_authoritative_psi_hessian_directional_derivative(
                                family,
                                synced_states,
                                specs,
                                hyper_layout,
                                psi_workspace.as_deref(),
                                psi_global,
                                &e_a,
                                total,
                            )?;
                        if let (Some(hdot_a), Some(psi_hdot_a)) = (hdot_a, psi_hdot_a) {
                            if weights.is_none() {
                                weights =
                                    Some(plan.explicit_param_mixed_trace_weights(pert_info)?);
                            }
                            let prepared =
                                weights.as_ref().expect("weights prepared on this axis");
                            let phi_psi_beta_a = prepared.contract(&hdot_a, &psi_hdot_a)?;
                            g[a_idx] -= phi_psi_beta_a;
                        }
                    }
                }
            }
        }
        let ld_s = match (s_logdet_blocks, penalty_motion.as_ref()) {
            (Some(blocks), Some((block_idx, _, _, s_psi_local))) => {
                blocks[*block_idx].tau_gradient_component(s_psi_local)
            }
            _ => 0.0,
        };

        // Explicit ψ motion of the Jeffreys curvature. This is a dense
        // correction even when the likelihood's ψ Hessian drift is carried by
        // an operator, so build it from the unpenalized information derivative
        // and compose it onto either representation below. The old code formed
        // this only in the dense branch and used the penalty-augmented `dense_b`
        // as ∂ψH_info; operator-backed spatial axes therefore omitted the term
        // entirely, while dense axes could contaminate it with ∂ψS.
        let explicit_jeffreys_hphi = if let (Some((z_j, h_joint)), Some(pert_info)) = (
            jeffreys_hphi_ctx
                .as_ref()
                .filter(|_| jeffreys_info_depends_on_psi),
            firth_pert_info.as_ref(),
        ) {
            match jeffreys_hphi_base.as_ref() {
                Some(base) => Some(base.perturbation_derivative_batched_axes(
                    pert_info,
                    firth_pert_axis_derivatives,
                )?),
                None => Some(
                    gam_solve::estimate::reml::jeffreys_subspace::joint_jeffreys_hphi_explicit_param_derivative(
                        h_joint.view(),
                        z_j.view(),
                        pert_info,
                        |dir: &Array1<f64>| {
                            family.joint_jeffreys_information_directional_derivative_with_specs(
                                synced_states,
                                specs,
                                dir,
                            )
                        },
                        |dir: &Array1<f64>| {
                            // Display boundary (gam#2689): the gam-solve
                            // explicit-parameter-derivative entry point takes
                            // `String`-erroring probes.
                            materialize_authoritative_psi_hessian_directional_derivative(
                                family,
                                synced_states,
                                specs,
                                hyper_layout,
                                psi_workspace.as_deref(),
                                psi_global,
                                dir,
                                total,
                            )
                            .map_err(|error| error.to_string())
                        },
                    )?,
                ),
            }
        } else {
            None
        };

        // Build drift: use block-local representation when possible to avoid
        // materializing full p×p dense matrices.
        let drift = if let Some(operator) = psi_terms.hessian_psi_operator {
            let mut drift = if let Some((_, start, end, s_psi_local)) = penalty_motion {
                // No dense Hessian contribution — penalty is block-local, operator
                // (if present) handles the likelihood part. O(p_block²) fast path.
                HyperCoordDrift::from_block_local_and_operator(
                    s_psi_local,
                    start,
                    end,
                    total,
                    Some(operator),
                )
            } else {
                HyperCoordDrift::from_parts(None, Some(operator))
            };
            drift.dense = explicit_jeffreys_hphi;
            drift
        } else {
            // Dense Hessian term exists (e.g., from non-implicit family).
            // Add block-local penalty motion only for DesignPenalty axes.
            let mut dense_b = psi_terms.hessian_psi;
            if let Some((_, start, end, s_psi_local)) = penalty_motion {
                dense_b
                    .slice_mut(ndarray::s![start..end, start..end])
                    .scaled_add(1.0, &s_psi_local);
            }
            if let Some(explicit_hphi) = explicit_jeffreys_hphi {
                dense_b += &explicit_hphi;
            }
            HyperCoordDrift::from_parts(Some(dense_b), None)
        };

        coords.push(HyperCoord {
            a,
            g,
            drift,
            ld_s,
            b_depends_on_beta: !hessian_beta_independent,
            is_penalty_like: false,
            firth_g: None,
            tk_eta_fixed: None,
            tk_x_fixed: None,
        });
    }

    log::info!(
        "[STAGE] build_psi_hyper_coords axis_count={} workspace_present={} elapsed={:.3}s",
        total_axes,
        psi_workspace.is_some(),
        build_psi_hyper_coords_start.elapsed().as_secs_f64(),
    );

    Ok(coords)
}

/// Build the direction-contracted ψψ second-order hook for the profiled θ-HVP
/// (#740).
///
/// Returns `Some(hook)` only when the family's psi workspace supplies a
/// combined-direction likelihood kernel (`second_order_terms_contracted`) that
/// covers every ψ basis axis; otherwise `None`, which keeps the outer-Hessian
/// operator on the exact per-pair `ext_ext_fn` assembly.
///
/// The hook produces, for the ψ-direction weights `α_ψ`, the
/// [`ContractedPsiSecondOrder`] ψψ-block contraction: it sums the family
/// likelihood contraction (from the workspace) with the generic ψψ penalty
/// motion, mirroring exactly the `α`-contraction of the per-pair `ext_ext`
/// callback's penalty terms (`½βᵀS_{ψiψj}β` into `objective`, `S_{ψiψj}β` into
/// `score`, `S_{ψiψj}` as a `BlockLocalDrift` into `hessian`, and the
/// `tau_hessian_component` into `ld_s`). Same-block-only, matching `ext_ext`.
///
/// `pub(crate)` so the #740 in-crate gate
/// `bernoulli_contracted_psi_hook_matches_per_pair_with_penalty` can assert the
/// generic penalty fold here equals `Σ_j α_j · build_psi_pair_callbacks().ext_ext(i, j)`.
///
/// Build the `(Z_J, H_info)` joint-Jeffreys/Firth context (gam#1607), mirroring
/// the inline construction in [`build_psi_hyper_coords`]. Returns `None` unless
/// the family uses the joint-Jeffreys term and exposes a dense joint Hessian, so
/// every non-Jeffreys / operator-only family is byte-unchanged.
/// The snapshot Jeffreys spectrum together with the per-ψ-axis information
/// derivatives `∂_{ψ_a}H_info|_β` and the ambient trace weights they induce.
///
/// The ψψ value second derivative `∂_{ψ_i}∂_{ψ_j}Φ` is `weights(ψ_i)` contracted
/// with `(∂_{ψ_j}H_info, ∂_{ψ_i}∂_{ψ_j}H_info)`. Both the spectrum and the
/// weights of a given first leg are fixed for the whole evaluation: the spectrum
/// by `(H_info, Z_J)` and the weights additionally by `∂_{ψ_i}H_info`, none of
/// which move with the second leg. gam#979: forming them inside the pair loop
/// cost one reduced-information eigendecomposition per PAIR — `axes²` of them
/// for one spectrum — which is what this cache removes. Each axis's weights are
/// prepared on first use, so an evaluation that never reaches an axis keeps its
/// exact previous outcome, including a stratum refusal it never provoked.
pub struct JeffreysPsiWeightCache {
    plan: gam_solve::estimate::reml::jeffreys_subspace::JointJeffreysPlan,
    /// `∂_{ψ_a}H_info|_β` for every ψ axis, in layout order.
    pub pert_first: Vec<Array2<f64>>,
    weights: Vec<
        std::sync::OnceLock<
            gam_solve::estimate::reml::jeffreys_subspace::JointJeffreysExplicitMixedTraceWeights,
        >,
    >,
}

impl JeffreysPsiWeightCache {
    /// Prepare the snapshot spectrum once and retain the axis derivatives.
    pub fn new(
        z_j: &Array2<f64>,
        h_joint: &Array2<f64>,
        pert_first: Vec<Array2<f64>>,
    ) -> Result<Self, CustomFamilyError> {
        let plan = gam_solve::estimate::reml::jeffreys_subspace::JointJeffreysPlan::prepare(
            h_joint.view(),
            z_j.view(),
        )?;
        let weights = (0..pert_first.len())
            .map(|_| std::sync::OnceLock::new())
            .collect();
        Ok(Self {
            plan,
            pert_first,
            weights,
        })
    }

    /// Number of ψ axes the cache carries derivatives for.
    #[must_use]
    pub fn axes(&self) -> usize {
        self.pert_first.len()
    }

    /// The ambient trace weights for first leg `axis`, prepared on first use.
    ///
    /// A concurrent first use recomputes rather than blocks; the result is a
    /// pure function of the plan and the axis derivative, so either writer
    /// stores the same weights.
    pub fn weights_for_axis(
        &self,
        axis: usize,
    ) -> Result<
        &gam_solve::estimate::reml::jeffreys_subspace::JointJeffreysExplicitMixedTraceWeights,
        CustomFamilyError,
    > {
        let slot = self.weights.get(axis).ok_or_else(|| {
            CustomFamilyError::trial_point(format!(
                "Jeffreys ψ weight cache has {} axes, asked for axis {axis}",
                self.weights.len()
            ))
        })?;
        if let Some(prepared) = slot.get() {
            return Ok(prepared);
        }
        let prepared = self
            .plan
            .explicit_param_mixed_trace_weights(&self.pert_first[axis])?;
        // `set` hands the weights back when a concurrent first use filled the
        // slot first. They are a pure function of the plan and this axis's
        // derivative, so whose copy the slot holds does not matter: the loser's
        // is dropped and every caller reads the stored one.
        if let Err(duplicate) = slot.set(prepared) {
            drop(duplicate);
        }
        Ok(slot.get().expect("weights stored for this axis"))
    }
}

pub fn build_jeffreys_hphi_ctx<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    synced_states: &[ParameterBlockState],
    specs: &[ParameterBlockSpec],
    hyper_layout: &CustomFamilyHyperLayout,
    total: usize,
) -> Result<Option<(Array2<f64>, Array2<f64>)>, CustomFamilyError> {
    if family.joint_jeffreys_term_required() && !hyper_layout.is_empty() {
        let ranges = block_param_ranges(specs);
        Ok(
            match (
                build_joint_jeffreys_subspace(family, specs, &ranges)?,
                family.joint_jeffreys_information_with_specs(synced_states, specs)?,
            ) {
                (Some(z), Some(h))
                    if z.nrows() == total && h.nrows() == total && h.ncols() == total =>
                {
                    Some((z, h))
                }
                _ => None,
            },
        )
    } else {
        Ok(None)
    }
}

pub fn build_contracted_psi_hook(
    specs: &[ParameterBlockSpec],
    hyper_layout: SharedCustomFamilyHyperLayout,
    beta_flat: &Array1<f64>,
    rho: &[f64],
    penalty_counts: &[usize],
    s_logdet_blocks: Option<&[PenaltyPseudologdet]>,
    psi_workspace: Option<Arc<dyn ExactNewtonJointPsiWorkspace>>,
    jeffreys_ctx: Option<(Array2<f64>, Array2<f64>)>,
) -> Result<Option<ContractedPsiSecondOrderFn>, CustomFamilyError> {
    // The contraction is a representation/cost choice for the family likelihood
    // ψψ second-order; without a contracted family kernel there is nothing to
    // accelerate, so decline (the per-pair `ext_ext_fn` path stays).
    let Some(workspace) = psi_workspace else {
        return Ok(None);
    };

    let total = beta_flat.len();
    let ranges = block_param_ranges(specs);
    let per_block = Arc::new(split_log_lambdas(
        &Array1::from_vec(rho.to_vec()),
        penalty_counts,
    )?);
    let per_block_lambdas = Arc::new(
        exact_lambdas_by_block(&per_block, "contracted psi log strength")?,
    );
    let beta_arc = Arc::new(beta_flat.clone());
    let ranges_arc = Arc::new(ranges);
    let s_logdet_block_cache = Arc::new(s_logdet_blocks.map(|blocks| blocks.to_vec()));

    // ψ → (block, local) location and block-local S_ψ for every ψ axis, built
    // once. `s_local` (block-local S_ψ) is reused for the τ-Hessian and as the
    // first leg of the bilinear `tr(S⁺ S_ψi S⁺ S_ψj)` penalty-logdet term.
    struct DesignPsiAxis {
        pub(crate) block: usize,
        pub(crate) local: usize,
        pub(crate) start: usize,
        pub(crate) end: usize,
        pub(crate) s_psi_local: Array2<f64>,
    }
    let mut axes: Vec<Option<DesignPsiAxis>> = Vec::with_capacity(hyper_layout.len());
    for axis_idx in 0..hyper_layout.len() {
        if let Some((block_idx, local_idx, deriv)) = hyper_layout.design_derivative(axis_idx) {
            let (start, end) = ranges_arc[block_idx];
            let p_block = end - start;
            let s_psi_local =
                assemble_block_local_s_psi(deriv, &per_block_lambdas[block_idx], p_block);
            axes.push(Some(DesignPsiAxis {
                block: block_idx,
                local: local_idx,
                start,
                end,
                s_psi_local,
            }));
        } else {
            axes.push(None);
        }
    }
    let axes = Arc::new(axes);
    let psi_dim = hyper_layout.len();
    if psi_dim == 0 {
        return Ok(None);
    }

    for axis_idx in 0..psi_dim {
        let mut basis = vec![0.0; psi_dim];
        basis[axis_idx] = 1.0;
        let Some(terms) = workspace.second_order_terms_contracted(&basis)? else {
            log::info!(
                "[outer-hvp contracted-psi] declined: workspace does not cover psi basis axis {}",
                axis_idx
            );
            return Ok(None);
        };
        if terms.objective.len() != psi_dim
            || terms.score.nrows() != psi_dim
            || terms.score.ncols() != total
            || terms.hessian.len() != psi_dim
        {
            return Err(CustomFamilyError::trial_point(format!(
                "contracted ψψ hook basis probe shape mismatch at axis {axis_idx}: \
                 objective={}, score={}x{}, hessian={}, psi_dim={psi_dim}, beta_dim={total}",
                terms.objective.len(),
                terms.score.nrows(),
                terms.score.ncols(),
                terms.hessian.len(),
            )));
        }
    }

    let hyper_layout = Arc::clone(&hyper_layout);

    // EXPLICIT Firth/Jeffreys ψψ VALUE second derivative context (gam#1607). The
    // outer LAML cost folds `−Φ(β̂)` with `Φ = ½ log|Z_Jᵀ H_info Z_J|₊` (gated),
    // and for a ψ length-scale that reshapes the design `H_info` depends on ψ
    // EXPLICITLY. The outer-Hessian ψψ block therefore needs the explicit second
    // derivative `−∂²_ψΦ` folded into each per-direction `objective[i]` (the
    // companion to the value gradient term `−∂_ψΦ` wired in
    // `build_psi_hyper_coords`). The exact contracted form for output row `i` and
    // applied ψ-direction `ψ(α)` is `−∂_{ψ_i}∂_{ψ(α)}Φ`, which the validated
    // Daleckii–Krein helper computes bilinearly from the three perturbations
    // `∂_{ψ_i}H_info`, `∂_{ψ(α)}H_info = Σ_j α_j ∂_{ψ_j}H_info`, and
    // `∂_{ψ_i}∂_{ψ(α)}H_info` (the contracted likelihood `hessian[i]`). We
    // precompute the per-axis first derivatives `∂_{ψ_j}H_info` here (β-fixed,
    // data-only — no penalty drift, matching the unpenalized Jeffreys info).
    // `None` (no Jeffreys term, or first-order terms unavailable) leaves a clean
    // / well-conditioned fit byte-unchanged.
    let firth_ctx: Option<Arc<JeffreysPsiWeightCache>> =
        match jeffreys_ctx {
            Some((z_j, h_joint))
                if z_j.nrows() == total && h_joint.nrows() == total && h_joint.ncols() == total =>
            {
                let first_terms: Option<Vec<ExactNewtonJointPsiTerms>> =
                    match workspace.first_order_terms_all()? {
                        Some(all) if all.len() == psi_dim => Some(all),
                        _ => {
                            let mut per_axis = Vec::with_capacity(psi_dim);
                            let mut ok = true;
                            for j in 0..psi_dim {
                                match workspace.first_order_terms(j)? {
                                    Some(t) => per_axis.push(t),
                                    None => {
                                        ok = false;
                                        break;
                                    }
                                }
                            }
                            if ok { Some(per_axis) } else { None }
                        }
                    };
                match first_terms {
                    Some(terms) => {
                        let mut pert_first: Vec<Array2<f64>> = Vec::with_capacity(psi_dim);
                        let mut ok = true;
                        for t in &terms {
                            if let Some(op) = t.hessian_psi_operator.as_ref() {
                                pert_first.push(op.mul_mat(&Array2::<f64>::eye(total)));
                            } else if t.hessian_psi.nrows() == total
                                && t.hessian_psi.ncols() == total
                            {
                                pert_first.push(t.hessian_psi.clone());
                            } else {
                                ok = false;
                                break;
                            }
                        }
                        if ok {
                            Some(Arc::new(JeffreysPsiWeightCache::new(
                                &z_j, &h_joint, pert_first,
                            )?))
                        } else {
                            None
                        }
                    }
                    None => None,
                }
            }
            _ => None,
        };

    let hook = move |alpha_psi: &[f64]| -> Result<Option<ContractedPsiSecondOrder>, CustomFamilyError> {
        if alpha_psi.len() != psi_dim {
            return Err(CustomFamilyError::trial_point(format!(
                "contracted ψψ hook: alpha_psi length {} != psi_dim {psi_dim}",
                alpha_psi.len()
            )));
        }
        // Family likelihood ψψ contraction (one combined-direction row pass).
        // The basis-axis probe above rejects partial kernels before the operator
        // skips per-pair ψψ tables; a decline here means the workspace violated
        // that coverage contract for a combined direction.
        let Some(likelihood) = workspace.second_order_terms_contracted(alpha_psi)? else {
            return Ok(None);
        };
        let mut objective = likelihood.objective;
        let mut score = likelihood.score;
        let mut ld_s = Array1::<f64>::zeros(psi_dim);
        // Per-output-row penalty drift `Σ_j α_j S_{ψi ψj}` (block-local),
        // composed onto the likelihood `hessian[i]` operator below.
        let mut hessian: Vec<DriftDerivResult> = likelihood.hessian;
        if objective.len() != psi_dim
            || score.nrows() != psi_dim
            || score.ncols() != total
            || hessian.len() != psi_dim
        {
            return Err(CustomFamilyError::trial_point(format!(
                "contracted ψψ hook: family kernel shape mismatch (objective={}, score={}x{}, hessian={}, psi_dim={psi_dim}, beta_dim={total})",
                objective.len(),
                score.nrows(),
                score.ncols(),
                hessian.len(),
            )));
        }

        for i in 0..psi_dim {
            // EXPLICIT Firth/Jeffreys ψψ VALUE second derivative (gam#1607):
            //   objective[i] -= ∂_{ψ_i}∂_{ψ(α)}Φ.
            // This applies to both design/penalty and family-owned axes.
            if let Some(jeffreys) = firth_ctx.as_ref() {
                let pert_i_alpha = match &hessian[i] {
                    DriftDerivResult::Dense(m) => m.clone(),
                    DriftDerivResult::Operator(op) => op.mul_mat(&Array2::<f64>::eye(total)),
                };
                let mut pert_alpha = Array2::<f64>::zeros((total, total));
                for (j, &aj) in alpha_psi.iter().enumerate() {
                    if aj != 0.0 {
                        pert_alpha.scaled_add(aj, &jeffreys.pert_first[j]);
                    }
                }
                let phi_psi_psi = jeffreys
                    .weights_for_axis(i)?
                    .contract(&pert_alpha, &pert_i_alpha)?;
                objective[i] -= phi_psi_psi;
            }

            // Family-owned axes have no generic penalty motion. Their complete
            // V_ij/g_ij/H_ij contribution is already in the workspace result.
            let Some(axis_i) = axes[i].as_ref() else {
                continue;
            };
            let p_block = axis_i.end - axis_i.start;
            let beta_block = beta_arc.slice(s![axis_i.start..axis_i.end]).to_owned();
            // Combined same-block penalty second derivative
            //   S_{ψi ψ(α)}_local = Σ_{j: block_j == block_i} α_j S_{ψi ψj}_local,
            // and the combined first-leg penalty derivative
            //   S_ψ(α)_local = Σ_{j: block_j == block_i} α_j S_ψj_local
            // (the second leg of the bilinear penalty-logdet cross term).
            let mut s_psi_psi_alpha = Array2::<f64>::zeros((p_block, p_block));
            let mut s_psi_alpha = Array2::<f64>::zeros((p_block, p_block));
            for (j, axis_j) in axes.iter().enumerate() {
                let Some(axis_j) = axis_j.as_ref() else {
                    continue;
                };
                let aj = alpha_psi[j];
                if aj == 0.0 || axis_j.block != axis_i.block {
                    continue;
                }
                let deriv_i = &hyper_layout.design_derivative_blocks()[axis_i.block][axis_i.local];
                let s_ij = assemble_block_local_s_psi_psi(
                    deriv_i,
                    axis_j.local,
                    &per_block_lambdas[axis_i.block],
                    p_block,
                );
                s_psi_psi_alpha.scaled_add(aj, &s_ij);
                s_psi_alpha.scaled_add(aj, &axis_j.s_psi_local);
            }

            // objective += 0.5 βᵀ S_{ψi ψ(α)} β  (matches ext_ext `a`).
            let s_beta = s_psi_psi_alpha.dot(&beta_block);
            objective[i] += 0.5 * beta_block.dot(&s_beta);
            // score[i] (block-local slice) += S_{ψi ψ(α)} β  (matches ext_ext `g`).
            {
                let mut score_local = score.row_mut(i);
                let mut slot = score_local.slice_mut(s![axis_i.start..axis_i.end]);
                slot += &s_beta;
            }
            // hessian[i] += S_{ψi ψ(α)} as a block-local drift (matches the
            // ext_ext `b_operator` BlockLocalDrift composite).
            let block_drift: Arc<dyn HyperOperator> = Arc::new(BlockLocalDrift {
                local: s_psi_psi_alpha.clone(),
                start: axis_i.start,
                end: axis_i.end,
                total_dim: total,
            });
            let combined = match std::mem::replace(
                &mut hessian[i],
                DriftDerivResult::Operator(Arc::clone(&block_drift)),
            ) {
                DriftDerivResult::Operator(existing) => {
                    DriftDerivResult::Operator(Arc::new(CompositeHyperOperator {
                        dense: None,
                        operators: vec![existing, block_drift],
                        dim_hint: total,
                    }))
                }
                DriftDerivResult::Dense(dense) => {
                    DriftDerivResult::Operator(Arc::new(CompositeHyperOperator {
                        dense: Some(dense),
                        operators: vec![block_drift],
                        dim_hint: total,
                    }))
                }
            };
            hessian[i] = combined;

            // ld_s[i] += Σ_j α_j tau_hessian_component(S_ψi, S_ψj, S_{ψiψj})
            //         = tau_hessian_component(S_ψi, S_ψ(α), S_{ψi ψ(α)})
            // by the (linearity in the second leg + bilinearity of the cross)
            // of the τ-Hessian; matches the ext_ext `ld_s` contraction.
            if let Some(ref logdet_blocks) = *s_logdet_block_cache {
                let pld = &logdet_blocks[axis_i.block];
                ld_s[i] = pld.tau_hessian_component(
                    &axis_i.s_psi_local,
                    &s_psi_alpha,
                    Some(&s_psi_psi_alpha),
                );
            }
        }

        Ok(Some(ContractedPsiSecondOrder {
            objective,
            score,
            hessian,
            ld_s,
        }))
    };

    // Display boundary (gam#2689): `ContractedPsiSecondOrderFn` is a gam-problem
    // alias over `Result<_, String>`, so the hook stays typed throughout and is
    // rendered exactly here, once, instead of at every `?` inside it.
    Ok(Some(Arc::new(move |alpha_psi: &[f64]| {
        hook(alpha_psi).map_err(|error| error.to_string())
    }) as ContractedPsiSecondOrderFn))
}

/// Build pair callbacks for ψ-ψ and ρ-ψ Hessian entries.
///
/// Returns two closures:
///
/// 1. **ext-ext** `(psi_i, psi_j) -> Result<HyperCoordPair, CustomFamilyError>`: second-order
///    fixed-β objects for a pair of ψ coordinates.
///
/// 2. **rho-ext** `(rho_k, psi_j) -> Result<HyperCoordPair, CustomFamilyError>`: mixed second-order
///    fixed-β objects for a ρ-ψ pair.
///
/// The closures capture (via `Arc`) shared references to penalty derivatives,
/// family state, and the penalty pseudo-inverse needed for logdet terms.
///
/// # Arguments
///
/// * `family` - The custom family instance (must be `Send + Sync + 'static`).
/// * `synced_states` - Synchronized block states at the current inner mode.
/// * `specs` - Parameter block specifications.
/// * `hyper_layout` - Typed global non-rho coordinate layout.
/// * `beta_flat` - Flattened joint coefficient vector at the inner mode.
/// * `rho` - Current log-smoothing parameters (flat).
/// * `penalty_counts` - Number of penalties per block.
/// * `s_logdet_blocks` - Optional exact block-local pseudologdet eigenspaces.
pub fn build_psi_pair_callbacks<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    synced_states: &[ParameterBlockState],
    specs: &[ParameterBlockSpec],
    hyper_layout: SharedCustomFamilyHyperLayout,
    beta_flat: &Array1<f64>,
    rho: &[f64],
    penalty_counts: &[usize],
    s_logdet_blocks: Option<&[PenaltyPseudologdet]>,
    psi_workspace: Option<Arc<dyn ExactNewtonJointPsiWorkspace>>,
    jeffreys_ctx: Option<(Array2<f64>, Array2<f64>)>,
) -> Result<
    (
        Box<dyn Fn(usize, usize) -> Result<HyperCoordPair, CustomFamilyError> + Send + Sync>,
        Box<dyn Fn(usize, usize) -> Result<HyperCoordPair, CustomFamilyError> + Send + Sync>,
    ),
    CustomFamilyError,
> {
    // Precompute shared data into Arc-wrapped clones for the closures.
    let ranges = block_param_ranges(specs);
    let total = beta_flat.len();
    let per_block = Arc::new(split_log_lambdas(
        &Array1::from_vec(rho.to_vec()),
        penalty_counts,
    )?);
    let per_block_lambdas = Arc::new(
        exact_lambdas_by_block(&per_block, "psi-pair callback log strength")
            ?,
    );
    let specs_arc = Arc::new(specs.to_vec());
    let beta_arc = Arc::new(beta_flat.clone());
    let synced_arc = Arc::new(synced_states.to_vec());
    let ranges_arc = Arc::new(ranges);
    let family_arc = Arc::new(family.clone());

    let s_logdet_block_cache = Arc::new(s_logdet_blocks.map(|blocks| blocks.to_vec()));

    struct PsiPenaltyCacheEntry {
        pub(crate) block_idx: usize,
        pub(crate) local_idx: usize,
        pub(crate) start: usize,
        pub(crate) end: usize,
        /// Block-local S_ψ matrix, stored for use with `PenaltyPseudologdet` methods.
        pub(crate) s_local: Option<Array2<f64>>,
    }

    struct RhoPenaltyCacheEntry {
        pub(crate) block_idx: usize,
        pub(crate) penalty_idx: usize,
        pub(crate) start: usize,
        pub(crate) end: usize,
        /// Unscaled penalty matrix S_k for use with `PenaltyPseudologdet::rho_tau_hessian_component`.
        pub(crate) s_k_unscaled: Array2<f64>,
    }

    // Build the psi coordinate cache once. These block-local S_psi matrices are
    // reused by ψψ and ρψ callbacks, avoiding repeated assembly inside the
    // O(q²) ext-ext loop.
    let mut psi_penalty_cache: Vec<Option<PsiPenaltyCacheEntry>> =
        Vec::with_capacity(hyper_layout.len());
    for axis_idx in 0..hyper_layout.len() {
        if let Some((block_idx, local_idx, deriv)) = hyper_layout.design_derivative(axis_idx) {
            let (start, end) = ranges_arc[block_idx];
            let p_block = end - start;
            let s_local = assemble_block_local_s_psi(deriv, &per_block_lambdas[block_idx], p_block);
            // Store the block-local S_ψ matrix when penalty logdet is active;
            // PenaltyPseudologdet methods will handle pseudoinverse and leakage internally.
            let s_local_opt = if s_logdet_block_cache.is_some() {
                Some(s_local)
            } else {
                None
            };
            psi_penalty_cache.push(Some(PsiPenaltyCacheEntry {
                block_idx,
                local_idx,
                start,
                end,
                s_local: s_local_opt,
            }));
        } else {
            psi_penalty_cache.push(None);
        }
    }
    let psi_penalty_cache = Arc::new(psi_penalty_cache);

    // EXPLICIT Firth/Jeffreys ψψ VALUE second-derivative context (gam#1607). The
    // per-pair ext_ext `a` is the explicit β-fixed second derivative `∂_{ψ_i}∂_{ψ_j}V`,
    // so it must carry `−∂_{ψ_i}∂_{ψ_j}Φ` to stay the exact derivative of the
    // ψ-gradient term `−∂_{ψ_i}Φ` that `build_psi_hyper_coords` adds to each coord's
    // `a` (otherwise the outer Hessian diverges from the FD of the gradient on the
    // per-pair Hessian path — the contracted-hook path carries the matching term in
    // `build_contracted_psi_hook`). We precompute the per-axis first derivatives
    // `∂_{ψ_j}H_info` (β-fixed, data-only, NO penalty drift — the Jeffreys info is the
    // unpenalized data Hessian), keyed by global ψ axis in the SAME order the cache /
    // `build_psi_hyper_coords` use. `None` (no Jeffreys term, or a first-order axis
    // term that can't be materialized total×total — matching the gradient term's own
    // availability gate) leaves a clean / well-conditioned fit byte-unchanged.
    let firth_pair_ctx: Option<Arc<JeffreysPsiWeightCache>> =
        match jeffreys_ctx {
            Some((z_j, h_joint))
                if z_j.nrows() == total && h_joint.nrows() == total && h_joint.ncols() == total =>
            {
                let psi_dim = hyper_layout.len();
                let batched_first: Option<Vec<ExactNewtonJointPsiTerms>> =
                    match psi_workspace.as_ref() {
                        Some(ws) => ws.first_order_terms_all()?,
                        None => None,
                    };
                if let Some(all) = batched_first.as_ref()
                    && all.len() != psi_dim
                {
                    return Err(CustomFamilyError::trial_point(format!(
                        "custom-family hyper workspace returned {} first-order axes for layout length {psi_dim}",
                        all.len()
                    )));
                }
                let mut pert_first: Vec<Array2<f64>> = Vec::with_capacity(psi_dim);
                let mut ok = true;
                for axis in 0..psi_dim {
                    let terms = if let Some(all) = batched_first.as_ref() {
                        all[axis].clone()
                    } else if let Some(ws) = psi_workspace.as_ref() {
                        if let Some(t) = ws.first_order_terms(axis)? {
                            t
                        } else {
                            family
                                .exact_newton_joint_psi_terms(
                                    synced_states,
                                    specs,
                                    &hyper_layout,
                                    axis,
                                )?
                                .ok_or_else(|| {
                                    format!(
                                        "typed hyper axis {axis} has no exact first-order terms"
                                    )
                                })?
                        }
                    } else {
                        family
                            .exact_newton_joint_psi_terms(
                                synced_states,
                                specs,
                                &hyper_layout,
                                axis,
                            )?
                            .ok_or_else(|| {
                                format!("typed hyper axis {axis} has no exact first-order terms")
                            })?
                    };
                    if let Some(op) = terms.hessian_psi_operator.as_ref() {
                        pert_first.push(op.mul_mat(&Array2::<f64>::eye(total)));
                    } else if terms.hessian_psi.nrows() == total
                        && terms.hessian_psi.ncols() == total
                    {
                        pert_first.push(terms.hessian_psi.clone());
                    } else {
                        ok = false;
                        break;
                    }
                }
                if ok {
                    Some(Arc::new(JeffreysPsiWeightCache::new(
                        &z_j, &h_joint, pert_first,
                    )?))
                } else {
                    None
                }
            }
            _ => None,
        };

    // Admission before allocation: the ρ-penalty cache densifies every block
    // penalty and holds them for the whole outer optimization, so its
    // aggregate footprint is charged on the process-wide ledger up front and
    // stays reserved for exactly the cache's lifetime (the `Governed` wrapper
    // couples the reservation to the Vec). A refusal is typed evidence the
    // joint budget cannot fit this dense cache right now.
    let rho_penalty_cache_bytes: usize = penalty_counts
        .iter()
        .enumerate()
        .flat_map(|(block_idx, &count)| (0..count).map(move |penalty_idx| (block_idx, penalty_idx)))
        .map(|(block_idx, penalty_idx)| {
            let (nrows, ncols) = specs_arc[block_idx].penalties[penalty_idx].shape();
            nrows
                .saturating_mul(ncols)
                .saturating_mul(std::mem::size_of::<f64>())
        })
        .fold(0usize, usize::saturating_add);
    let rho_penalty_reservation = gam_runtime::resource::MemoryGovernor::global()
        .try_reserve(
            rho_penalty_cache_bytes,
            "custom_family::psi_hyper::rho_penalty_cache",
        )
        .map_err(|err| CustomFamilyError::trial_point(format!("rho-penalty dense cache refused by memory governor: {err}")))?;
    let mut rho_penalty_cache: Vec<RhoPenaltyCacheEntry> = Vec::new();
    for (block_idx, &count) in penalty_counts.iter().enumerate() {
        let (start, end) = ranges_arc[block_idx];
        for penalty_idx in 0..count {
            let s_k_unscaled = specs_arc[block_idx].penalties[penalty_idx].to_dense();
            rho_penalty_cache.push(RhoPenaltyCacheEntry {
                block_idx,
                penalty_idx,
                start,
                end,
                s_k_unscaled,
            });
        }
    }
    let rho_penalty_cache = Arc::new(rho_penalty_reservation.bind(rho_penalty_cache));

    // A family-owned coordinate changes likelihood geometry directly.  Every
    // pair touching one therefore requires explicit V_ij/g_ij/H_ij coverage;
    // treating a missing pair as zero would silently certify a different
    // objective.  Probe coverage while this constructor can still return a
    // typed error.  The immutable workspace may cache the corresponding row
    // program, so the callback's later lookup remains cheap.
    let mut family_pair_cache =
        vec![
            vec![None::<ExactNewtonJointPsiSecondOrderTerms>; hyper_layout.len()];
            hyper_layout.len()
        ];
    for i in 0..hyper_layout.len() {
        for j in i..hyper_layout.len() {
            if hyper_layout.family_axis(i).is_none() && hyper_layout.family_axis(j).is_none() {
                continue;
            }
            let pair = if let Some(workspace) = psi_workspace.as_ref() {
                workspace.second_order_terms(i, j)?
            } else {
                family.exact_newton_joint_psisecond_order_terms(
                    synced_states,
                    specs,
                    &hyper_layout,
                    i,
                    j,
                )?
            };
            let pair = pair.ok_or_else(|| {
                format!("typed family hyper pair ({i}, {j}) has no exact V_ij/g_ij/H_ij terms")
            })?;
            family_pair_cache[i][j] = Some(pair.clone());
            family_pair_cache[j][i] = Some(pair);
        }
    }
    let family_pair_cache = Arc::new(family_pair_cache);

    // ψ-ψ pair callback
    let ext_ext = {
        let per_block_lambdas = Arc::clone(&per_block_lambdas);
        let hyper_layout = Arc::clone(&hyper_layout);
        let specs_arc = Arc::clone(&specs_arc);
        let beta_arc = Arc::clone(&beta_arc);
        let synced_arc = Arc::clone(&synced_arc);
        let s_logdet_block_cache = Arc::clone(&s_logdet_block_cache);
        let psi_penalty_cache = Arc::clone(&psi_penalty_cache);
        let family_arc = Arc::clone(&family_arc);
        let psi_workspace = psi_workspace.clone();
        let firth_pair_ctx = firth_pair_ctx.clone();
        let family_pair_cache = Arc::clone(&family_pair_cache);

        Box::new(
            move |psi_i: usize, psi_j: usize| -> Result<HyperCoordPair, CustomFamilyError> {
                if psi_i >= hyper_layout.len() || psi_j >= hyper_layout.len() {
                    return Err(CustomFamilyError::trial_point(format!(
                        "typed hyper pair index out of bounds: ({psi_i}, {psi_j}) for {} axes",
                        hyper_layout.len()
                    )));
                }
                let cache_i = psi_penalty_cache[psi_i].as_ref();
                let cache_j = psi_penalty_cache[psi_j].as_ref();

                // Get family-provided second-order likelihood terms.
                let family_pair_required = hyper_layout.family_axis(psi_i).is_some()
                    || hyper_layout.family_axis(psi_j).is_some();
                let psi2 = if family_pair_required {
                    Some(family_pair_cache[psi_i][psi_j].clone().ok_or_else(|| {
                    format!(
                        "typed family hyper pair ({psi_i}, {psi_j}) was not retained by its validated pair cache"
                    )
                })?)
                } else {
                    let terms = if let Some(workspace) = psi_workspace.as_ref() {
                        workspace.second_order_terms(psi_i, psi_j)
                    } else {
                        family_arc.exact_newton_joint_psisecond_order_terms(
                            &synced_arc,
                            &specs_arc,
                            &hyper_layout,
                            psi_i,
                            psi_j,
                        )
                    };
                    terms.map_err(|error| {
                    format!(
                        "typed design hyper pair ({psi_i}, {psi_j}) failed during immutable Hessian assembly: {error}"
                    )
                })?
                };

                let (obj_ll, score_ll, hess_ll, hess_ll_op) = match psi2 {
                    Some(t) => (
                        t.objective_psi_psi,
                        t.score_psi_psi,
                        t.hessian_psi_psi,
                        t.hessian_psi_psi_operator,
                    ),
                    None => (
                        0.0,
                        Array1::zeros(total),
                        Array2::zeros((total, total)),
                        None,
                    ),
                };

                let mut a = obj_ll;
                let mut g = score_ll;
                let mut b_mat = hess_ll;
                let mut b_operator = hess_ll_op;
                if g.len() != total {
                    return Err(CustomFamilyError::trial_point(format!(
                        "typed hyper pair ({psi_i}, {psi_j}) returned score length {}, expected {total}",
                        g.len()
                    )));
                }
                if b_mat.dim() != (0, 0) && b_mat.dim() != (total, total) {
                    return Err(CustomFamilyError::trial_point(format!(
                        "typed hyper pair ({psi_i}, {psi_j}) returned dense Hessian shape {:?}, expected (0, 0) or ({total}, {total})",
                        b_mat.dim()
                    )));
                }

                // EXPLICIT Firth/Jeffreys ψψ VALUE second derivative (gam#1607):
                //   a -= ∂_{ψ_i}∂_{ψ_j}Φ
                // the per-pair companion to the gradient term `−∂_{ψ_i}Φ` added to each
                // coord's `a` in `build_psi_hyper_coords` (and to the contracted-hook
                // `objective[i]` in `build_contracted_psi_hook`). Computed from the
                // β-fixed perturbations `∂_{ψ_i}H_info` / `∂_{ψ_j}H_info` (pert_first) and
                // the UNPENALIZED mixed second `∂_{ψ_i}∂_{ψ_j}H_info` (`b_mat`/`b_operator`
                // captured HERE, before the `S_{ψ_i ψ_j}` penalty drift is folded in below
                // — the Jeffreys info is the unpenalized data Hessian). The helper returns
                // `0.0` when the conditioning gate skips the term, so a clean fit is
                // byte-unchanged. Invalid shape/eigensystem evidence propagates through
                // the pair callback together with workspace failures.
                if let Some(jeffreys) = firth_pair_ctx.as_ref()
                    && psi_i < jeffreys.axes()
                    && psi_j < jeffreys.axes()
                {
                    let pert_ij_opt: Option<Array2<f64>> =
                        if b_mat.nrows() == total && b_mat.ncols() == total {
                            Some(b_mat.clone())
                        } else {
                            b_operator
                                .as_ref()
                                .map(|op| op.mul_mat(&Array2::<f64>::eye(total)))
                        };
                    if let Some(pert_ij) = pert_ij_opt {
                        let phi_psi_psi = jeffreys
                            .weights_for_axis(psi_i)
                            .and_then(|weights| {
                                weights
                                    .contract(&jeffreys.pert_first[psi_j], &pert_ij)
                                    .map_err(CustomFamilyError::from)
                            })
                            .map_err(|error| {
                                format!(
                                    "typed hyper pair ({psi_i}, {psi_j}) Jeffreys second derivative failed: {error}"
                                )
                            })?;
                        a -= phi_psi_psi;
                    }
                }

                // Assemble S_{ψ_i ψ_j} only on the touched block.
                let ld_s = if let (Some(cache_i), Some(cache_j)) = (cache_i, cache_j)
                    && cache_i.block_idx == cache_j.block_idx
                {
                    let p_block = cache_i.end - cache_i.start;
                    let deriv_i = &hyper_layout.design_derivative_blocks()[cache_i.block_idx]
                        [cache_i.local_idx];
                    let s_local = assemble_block_local_s_psi_psi(
                        deriv_i,
                        cache_j.local_idx,
                        &per_block_lambdas[cache_i.block_idx],
                        p_block,
                    );

                    let beta_block = beta_arc.slice(s![cache_i.start..cache_i.end]).to_owned();
                    let s_ij_beta_local = s_local.dot(&beta_block);
                    a += 0.5 * beta_block.dot(&s_ij_beta_local);
                    {
                        let mut g_local = g.slice_mut(s![cache_i.start..cache_i.end]);
                        g_local += &s_ij_beta_local;
                    }
                    // The S_{ψ_i ψ_j} block contribution attaches to the dense
                    // Hessian when the family returned a dense `b_mat`, and to
                    // the operator-backed Hessian (via a `BlockLocalDrift`
                    // composite) when the family returned `hessian_psi_psi`
                    // empty alongside an operator. Slicing into a `(0, 0)`
                    // dense matrix would otherwise panic in the matrix-free
                    // path that survival-marginal-slope and other operator-
                    // backed families use.
                    if b_mat.nrows() > 0 {
                        let mut b_local = b_mat
                            .slice_mut(s![cache_i.start..cache_i.end, cache_i.start..cache_i.end]);
                        b_local += &s_local;
                    } else {
                        let block_drift: Arc<dyn HyperOperator> = Arc::new(BlockLocalDrift {
                            local: s_local.clone(),
                            start: cache_i.start,
                            end: cache_i.end,
                            total_dim: total,
                        });
                        b_operator = Some(match b_operator.take() {
                            Some(existing) => Arc::new(CompositeHyperOperator {
                                dense: None,
                                operators: vec![existing, block_drift],
                                dim_hint: total,
                            })
                                as Arc<dyn HyperOperator>,
                            None => block_drift,
                        });
                    }

                    if let Some(ref logdet_blocks) = *s_logdet_block_cache {
                        let pld = &logdet_blocks[cache_i.block_idx];
                        let s_psi_i = cache_i
                        .s_local
                        .as_ref()
                        .ok_or_else(|| {
                            format!(
                                "typed hyper axis {psi_i} has no cached S_psi for active penalty logdet"
                            )
                        })?;
                        let s_psi_j = cache_j
                        .s_local
                        .as_ref()
                        .ok_or_else(|| {
                            format!(
                                "typed hyper axis {psi_j} has no cached S_psi for active penalty logdet"
                            )
                        })?;
                        // τ-Hessian: tr(S⁺ S_{ψi ψj}) − tr(S⁺ S_ψi S⁺ S_ψj) + 2 tr(Σ₊⁻² L_i L_j^T)
                        pld.tau_hessian_component(s_psi_i, s_psi_j, Some(&s_local))
                    } else {
                        0.0
                    }
                } else {
                    0.0
                };

                Ok(HyperCoordPair {
                    a,
                    g,
                    b_mat,
                    b_operator,
                    ld_s,
                })
            },
        ) as Box<dyn Fn(usize, usize) -> Result<HyperCoordPair, CustomFamilyError> + Send + Sync>
    };

    // ρ-ψ pair callback
    let rho_ext = {
        let per_block_lambdas = Arc::clone(&per_block_lambdas);
        let hyper_layout = Arc::clone(&hyper_layout);
        let beta_arc = Arc::clone(&beta_arc);
        let psi_penalty_cache = Arc::clone(&psi_penalty_cache);
        let rho_penalty_cache = Arc::clone(&rho_penalty_cache);
        let s_logdet_block_cache = Arc::clone(&s_logdet_block_cache);

        Box::new(
            move |rho_k: usize, psi_j: usize| -> Result<HyperCoordPair, CustomFamilyError> {
                if rho_k >= rho_penalty_cache.len() || psi_j >= hyper_layout.len() {
                    return Err(CustomFamilyError::trial_point(format!(
                        "rho×typed-hyper pair index out of bounds: ({rho_k}, {psi_j}) for {} rho and {} non-rho axes",
                        rho_penalty_cache.len(),
                        hyper_layout.len()
                    )));
                }
                let rho_cache = &rho_penalty_cache[rho_k];
                let psi_cache = psi_penalty_cache[psi_j].as_ref();
                let mut a = 0.0;
                let mut g = Array1::<f64>::zeros(total);
                let mut b_mat = Array2::<f64>::zeros((total, total));

                // S_{ρ_k, ψ_j} = λ_k ∂S_k/∂ψ_j.
                // Only nonzero when both coordinates share the same block and the
                // ψ derivative touches the k-th penalty.
                let ld_s = if let Some(psi_cache) = psi_cache
                    && rho_cache.block_idx == psi_cache.block_idx
                {
                    let p_block = rho_cache.end - rho_cache.start;
                    let deriv = &hyper_layout.design_derivative_blocks()[psi_cache.block_idx]
                        [psi_cache.local_idx];
                    let lambda_k = per_block_lambdas[rho_cache.block_idx][rho_cache.penalty_idx];
                    let local = if let Some(ref components) = deriv.s_psi_penalty_components {
                        let mut m = Array2::<f64>::zeros((p_block, p_block));
                        for (penalty_idx, s_part) in components {
                            if *penalty_idx == rho_cache.penalty_idx {
                                s_part.add_scaled_to(lambda_k, &mut m);
                            }
                        }
                        m
                    } else if let Some(ref components) = deriv.s_psi_components {
                        let mut m = Array2::<f64>::zeros((p_block, p_block));
                        for (penalty_idx, s_part) in components {
                            if *penalty_idx == rho_cache.penalty_idx {
                                m.scaled_add(lambda_k, s_part);
                            }
                        }
                        m
                    } else if deriv.penalty_index == Some(rho_cache.penalty_idx) {
                        deriv.s_psi.mapv(|v| lambda_k * v)
                    } else {
                        Array2::<f64>::zeros((p_block, p_block))
                    };

                    let beta_block = beta_arc
                        .slice(s![rho_cache.start..rho_cache.end])
                        .to_owned();
                    let s_kj_beta_local = local.dot(&beta_block);
                    a = 0.5 * beta_block.dot(&s_kj_beta_local);
                    {
                        let mut g_local = g.slice_mut(s![rho_cache.start..rho_cache.end]);
                        g_local += &s_kj_beta_local;
                    }
                    {
                        let mut b_local = b_mat.slice_mut(s![
                            rho_cache.start..rho_cache.end,
                            rho_cache.start..rho_cache.end
                        ]);
                        b_local += &local;
                    }

                    if let Some(ref logdet_blocks) = *s_logdet_block_cache {
                        let pld = &logdet_blocks[rho_cache.block_idx];
                        let s_psi_j = psi_cache
                        .s_local
                        .as_ref()
                        .ok_or_else(|| {
                            format!(
                                "typed hyper axis {psi_j} has no cached S_psi for active penalty logdet"
                            )
                        })?;
                        // ∂S_k/∂ψ_j (unscaled): extract from local by dividing out λ_k.
                        // A strength is `exp(ρ)`, positive unless it has underflowed to
                        // exactly zero; that is the one case with no `S_k/λ_k`.
                        let ds_k_dpsi = if lambda_k != 0.0 {
                            Some(local.mapv(|v| v / lambda_k))
                        } else {
                            None
                        };
                        // Mixed ρ×τ Hessian: λ_k [tr(S⁺ ∂S_k/∂ψ_j) − tr(S⁺ S_k S⁺ S_ψj)]
                        pld.rho_tau_hessian_component(
                            &rho_cache.s_k_unscaled,
                            lambda_k,
                            s_psi_j,
                            ds_k_dpsi.as_ref(),
                        )
                    } else {
                        0.0
                    }
                } else {
                    0.0
                };

                Ok(HyperCoordPair {
                    a,
                    g,
                    b_mat,
                    b_operator: None,
                    ld_s,
                })
            },
        ) as Box<dyn Fn(usize, usize) -> Result<HyperCoordPair, CustomFamilyError> + Send + Sync>
    };

    Ok((ext_ext, rho_ext))
}

/// Build the `M_i\[u\] = D_β B_i\[u\]` callback for ψ coordinates.
///
/// This wraps `family.exact_newton_joint_psihessian_directional_derivative`
/// into the unified `FixedDriftDerivFn` signature. For each external
/// (ψ) coordinate index `ext_idx`, calling `f(ext_idx, &direction)` returns
/// `Some(D_β H_ψ[u])` when the family provides it, or `None` otherwise.
///
/// The returned closure also adds the penalty-side β-drift when the ψ
/// coordinate moves realized penalties: `D_β S_ψ[u] = 0` for ψ that
/// only enters via the likelihood, so the penalty contribution vanishes
/// and the callback delegates entirely to the family hook. (Penalty
/// matrices S_ψ do not depend on β, so their β-directional derivative
/// is zero.)
///
/// # Returns
///
/// `Some(callback)` when the family potentially provides the drift term.
/// `None` when the family is Gaussian (B_i is β-independent for all
/// coordinates, so M_i ≡ 0).
pub fn build_psi_drift_deriv_callback<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    synced_states: &[ParameterBlockState],
    specs: &[ParameterBlockSpec],
    hyper_layout: SharedCustomFamilyHyperLayout,
    hessian_beta_independent: bool,
    psi_workspace: Option<Arc<dyn ExactNewtonJointPsiWorkspace>>,
) -> Result<Option<FixedDriftDerivFn>, CustomFamilyError> {
    if hessian_beta_independent {
        // Likelihood Hessian is β-independent; M_i ≡ 0.
        return Ok(None);
    }

    if hyper_layout.family_axis_count() != 0 && psi_workspace.is_none() {
        return Err(CustomFamilyError::trial_point(
            "family-owned hyper axes require one owned exact-psi workspace for directional Hessian drift",
        ));
    }

    let synced_arc = Arc::new(synced_states.to_vec());
    let specs_arc = Arc::new(specs.to_vec());
    let family_arc = Arc::new(family.clone());
    let psi_workspace = psi_workspace;

    let typed_drift = move |ext_idx: usize,
                            direction: &Array1<f64>|
          -> Result<Option<DriftDerivResult>, CustomFamilyError> {
            // The family hook takes a psi index (0-based within ψ coordinates)
            // and a flattened coefficient direction.
            let result = if let Some(workspace) = psi_workspace.as_ref() {
                workspace.hessian_directional_derivative(ext_idx, direction)
            } else {
                family_arc
                    .exact_newton_joint_psihessian_directional_derivative(
                        &synced_arc,
                        &specs_arc,
                        &hyper_layout,
                        ext_idx,
                        direction,
                    )
                    .map(|drift| drift.map(DriftDerivResult::Dense))
            };
            match result? {
                Some(drift) => Ok(Some(drift)),
                None if hyper_layout.family_axis(ext_idx).is_some() => Err(CustomFamilyError::trial_point(format!(
                    "family-owned hyper axis {ext_idx} has no exact D_beta H_i[u] term for the requested direction"
                ))),
                None => Ok(None),
            }
    };
    // Display boundary (gam#2689): `FixedDriftDerivFn` is a gam-problem alias
    // over `Result<_, String>`; the drift closure stays typed and is rendered
    // exactly once, here.
    Ok(Some(Box::new(
        move |ext_idx: usize, direction: &Array1<f64>| {
            typed_drift(ext_idx, direction).map_err(|error| error.to_string())
        },
    )))
}

pub(crate) fn evaluate_custom_family_hyper_internal<
    F: CustomFamily + Clone + Send + Sync + 'static,
>(
    family: &F,
    specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
    penalty_counts: &[usize],
    rho_current: &Array1<f64>,
    hyper_layout: &CustomFamilyHyperLayout,
    warm_start: Option<&ConstrainedWarmStart>,
    rho_prior: gam_problem::RhoPrior,
    eval_mode: EvalMode,
) -> Result<OuterObjectiveEvalResult, CustomFamilyError> {
    evaluate_custom_family_hyper_internal_shared(
        family,
        specs,
        options,
        penalty_counts,
        rho_current,
        Arc::new(hyper_layout.clone()),
        warm_start,
        rho_prior,
        eval_mode,
        eval_mode,
        None,
    )
}

/// Whether an inner solve that missed the tightened derivative-quality floor
/// nevertheless satisfies the tolerance its CALLER asked for.
///
/// The tightening below (`JOINT_LAML_DERIV_INNER_TOL_FLOOR`) is an accuracy
/// REQUEST, not a feasibility precondition: it buys a `β̂` at which the
/// exact-Newton trace-gradient's `D_βH` coupling is exact, and the value-only
/// path is already evaluated at the caller's own `inner_tol` without anyone
/// calling that unsound. So a solve that reaches the caller's contract and no
/// further has produced exactly what the value path produces, and refusing it
/// converts "this seed's inner problem is not locally convex at a 1e9 penalty
/// scale" into "no candidate seeds passed outer startup validation" — measured
/// on the survival location-scale (gam#2695) and frailty (gam#2714) fits,
/// where every seed reached a relative stationarity of `1e-9`-`1e-8` against a
/// caller tolerance of `1e-6` and was refused for missing `1e-11`, so the outer
/// search never took its first step.
///
/// The comparison is the one the solver itself makes: `residual ≤ inner_tol ·
/// (1 + stationarity_scale)`, reconstructed from the terminal state's own
/// `stationarity_scale` so the two sides cannot drift apart. Only the joint
/// Newton path is judged here — it is the only one the floor tightens, and it
/// is the only terminal state that carries a residual to judge.
pub(crate) fn joint_newton_meets_caller_inner_contract(
    terminal: Option<&gam_problem::InnerConvergenceTerminalState>,
    caller_inner_tol: f64,
) -> bool {
    let Some(gam_problem::InnerConvergenceTerminalState::JointNewton {
        stationarity_residual,
        stationarity_scale,
        ..
    }) = terminal
    else {
        return false;
    };
    if !(caller_inner_tol.is_finite() && caller_inner_tol > 0.0) {
        return false;
    }
    if !stationarity_residual.is_finite() || !stationarity_scale.is_finite() {
        return false;
    }
    *stationarity_residual <= caller_inner_tol * (1.0 + stationarity_scale.abs())
}

/// Evaluate the rho-only Laplace criterion from a coefficient mode this caller
/// already owns.
///
/// Continuation correctors deliberately produce no determinant artifacts. The
/// endpoint, however, is judged by the same complete criterion as every normal
/// value-only outer evaluation. Consuming the owned mode here avoids a second
/// coefficient solve while leaving the authoritative joint outer evaluator in
/// sole ownership of the endpoint logdet and prior scalar.
pub(crate) fn evaluate_custom_family_hyper_from_coefficient_mode<
    F: CustomFamily + Clone + Send + Sync + 'static,
>(
    family: &F,
    specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
    penalty_counts: &[usize],
    rho_current: &Array1<f64>,
    rho_prior: gam_problem::RhoPrior,
    inner: BlockwiseInnerResult,
) -> Result<OuterObjectiveEvalResult, CustomFamilyError> {
    let hyper_layout = CustomFamilyHyperLayout::new(
        vec![Vec::<CustomFamilyBlockPsiDerivative>::new(); specs.len()],
        Vec::new(),
        Array1::zeros(0),
    )?;
    evaluate_custom_family_hyper_internal_shared(
        family,
        specs,
        options,
        penalty_counts,
        rho_current,
        Arc::new(hyper_layout),
        None,
        rho_prior,
        EvalMode::ValueOnly,
        EvalMode::ValueOnly,
        Some(inner),
    )
}

fn evaluate_custom_family_hyper_internal_shared<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
    penalty_counts: &[usize],
    rho_current: &Array1<f64>,
    hyper_layout: SharedCustomFamilyHyperLayout,
    warm_start: Option<&ConstrainedWarmStart>,
    rho_prior: gam_problem::RhoPrior,
    eval_mode: EvalMode,
    inner_quality_mode: EvalMode,
    precomputed_inner: Option<BlockwiseInnerResult>,
) -> Result<OuterObjectiveEvalResult, CustomFamilyError> {
    if hyper_layout.block_count() != specs.len() {
        crate::bail_dim_custom!(
            "joint hyper layout block count mismatch: got {}, expected {}",
            hyper_layout.block_count(),
            specs.len()
        );
    }

    if penalty_counts.len() != specs.len() {
        crate::bail_dim_custom!(
            "joint hyper penalty-count block mismatch: got {}, expected {}",
            penalty_counts.len(),
            specs.len()
        );
    }
    let rho_dim = penalty_counts.iter().sum::<usize>();
    let psi_dim = hyper_layout.len();
    if rho_current.len() != rho_dim {
        crate::bail_dim_custom!(
            "joint hyper rho dimension mismatch: got {}, expected {} (psi={})",
            rho_current.len(),
            rho_dim,
            psi_dim
        );
    }

    // ── Common setup: inner solve, ridge, refresh, ranges ──
    let include_logdet_h = include_exact_newton_logdet_h(family, options);
    let include_logdet_s = include_exact_newton_logdet_s(family, options);
    let strict_spd = use_exact_newton_strict_spd(family);
    let per_block = split_log_lambdas(rho_current, penalty_counts)?;
    let psi_safe_warm_start =
        warm_start_without_cached_inner_for_psi_derivatives(warm_start, psi_dim > 0);

    // gam#1820: for a COUPLED family whose joint Hessian depends on β, the
    // exact-Newton LAML outer GRADIENT `½tr(H⁻¹Ḣ)` — including its `D_βH[β_i]`
    // mode-response coupling across blocks — and the inner KKT-residual
    // correction are only mutually consistent at a JOINT-stationary β̂. The
    // inner solve certifies joint stationarity only to `inner_tol` (default
    // 1e-6 ⇒ ‖r‖≈3e-9; a deliberately loose 1e-3 ⇒ ‖r‖≈1.5e-4), and that
    // residual desyncs the analytic trace-gradient from all three autodiff
    // engines / the joint-stationarity requirement. The joint-Newton mode
    // converges quadratically, so tightening the derivative-path inner solve to
    // a stationarity floor costs ~one extra step while pinning β̂ at the true
    // optimum where the trace-gradient's block-coupled `D_βH` term is exact.
    // Value-only line-search evaluations normally keep the caller's tolerance.
    // Atomic multi-start screening supplies the requested derivative mode as
    // `inner_quality_mode`, so the objective winner is solved once at the exact
    // quality its derivatives require and that owned mode can be reused below.
    // Restricted to the ρ-only joint path (`psi_dim == 0`): ψ-bearing
    // evaluations already pass through the monotone caller-authority rule in
    // `derivative_quality_options_and_warm_start`; this local branch must not
    // impose a second, competing coefficient-quality policy.
    const JOINT_LAML_DERIV_INNER_TOL_FLOOR: f64 = 1e-11;
    let tighten_inner_for_deriv = psi_dim == 0
        && include_logdet_h
        && inner_quality_mode != EvalMode::ValueOnly
        && family.has_explicit_joint_hessian()
        && options.inner_tol > JOINT_LAML_DERIV_INNER_TOL_FLOOR;
    let tightened_options = tighten_inner_for_deriv.then(|| {
        let mut tightened = options.clone();
        tightened.inner_tol = JOINT_LAML_DERIV_INNER_TOL_FLOOR;
        tightened.inner_max_cycles = tightened.inner_max_cycles.max(200);
        tightened
    });
    let inner_solve_options = tightened_options.as_ref().unwrap_or(options);
    let mut inner = match precomputed_inner {
        Some(inner) => inner,
        None => inner_blockwise_fit(
            family,
            specs,
            &per_block,
            inner_solve_options,
            psi_safe_warm_start.as_ref().or(warm_start),
        )?,
    };
    // A solve held to the tightened floor that lands inside the caller's own
    // contract has met the contract; see
    // [`joint_newton_meets_caller_inner_contract`] for why the difference is a
    // quality request rather than a soundness condition.
    if !inner.converged
        && tightened_options.is_some()
        && joint_newton_meets_caller_inner_contract(
            inner.terminal_convergence_state.as_ref(),
            options.inner_tol,
        )
    {
        log::info!(
            "[OUTER] the derivative-mode inner solve missed the {:.1e} derivative-quality \
             floor but meets the caller's inner_tol={:.1e} on its own terminal state; \
             accepting it at the caller's contract, as the value-only path already is",
            JOINT_LAML_DERIV_INNER_TOL_FLOOR,
            options.inner_tol,
        );
        inner.converged = true;
    }
    if !inner.converged {
        let theta_dim = rho_dim + psi_dim;
        // #2553: the fact that matters is "this trial point is
        // infeasible", and it belongs in the variant rather than the
        // message. `UnsupportedConfiguration` MEANS the configuration is
        // structurally unsupported — a fatal claim — so encoding a
        // recoverable per-theta condition in it forced every downstream
        // consumer to recover the distinction from prose, and two of them
        // reached opposite verdicts.
        return Err(CustomFamilyError::InnerSolveNotConverged {
            cycles: inner.cycles,
            // Carry the quantity the verdict was taken on. `inner.kkt_residual`
            // is live here and was previously dropped, so the refusal could say
            // only how many cycles ran — which cannot tell a budget-limited
            // solve apart from a stalled one.
            kkt_residual: inner
                .kkt_residual
                .as_ref()
                .map(ProjectedKktResidual::inf_norm),
            kkt_tol: inner
                .kkt_residual
                .as_ref()
                .and_then(ProjectedKktResidual::residual_tol),
            // `kkt_residual` is `None` off a converged iterate BY DESIGN — no
            // caller may trust an IFT correction at a non-KKT point, so that
            // field stays empty here and cannot be the diagnostic. The decision
            // variables the loop's verdict was actually taken on can be
            // reported, and they are what separates "needs more cycles" from
            // "the exact joint stationarity gate is the blocker".
            terminal: inner.terminal_convergence_state,
            theta_dim,
            rho_dim,
            psi_dim,
        });
    }
    let ridge = effective_solverridge(options.ridge_floor);
    let moderidge = if options.ridge_policy.accounts_for_objective() {
        ridge
    } else {
        0.0
    };
    let extra_logdet_ridge = 0.0;

    refresh_all_block_etas(family, specs, &mut inner.block_states)?;
    let ranges = block_param_ranges(specs);
    let total = ranges.last().map(|(_, e)| *e).unwrap_or(0);
    // ── Try to obtain a joint Hessian and route through the unified evaluator ──
    //
    // When psi_dim > 0, exact Newton is required because the ψ derivative
    // callbacks use exact Newton trait methods. When psi_dim == 0,
    // build_joint_hessian_closures handles both exact Newton and surrogate.
    let cthf_internal_psi_branch_start = std::time::Instant::now();
    if psi_dim > 0 {
        log::info!(
            "[STAGE] cthf_internal psi_dim={} eval_mode={:?} pre_unified elapsed={:.3}s",
            psi_dim,
            eval_mode,
            cthf_internal_psi_branch_start.elapsed().as_secs_f64(),
        );
        // ψ coordinates present: require exact Newton Hessian for consistency
        // with the psi derivative callbacks.
        let beta_flat = flatten_state_betas(&inner.block_states, specs);
        let synced_joint_states = Arc::new(synchronized_states_from_flat_beta(
            family,
            specs,
            &inner.block_states,
            &beta_flat,
        )?);
        let hessian_workspace = match inner.joint_workspace.clone() {
            Some(workspace) => Some(workspace),
            None => family.exact_newton_joint_hessian_workspace_with_options(
                synced_joint_states.as_ref(),
                specs,
                options,
            )?,
        };
        // Outer-eval entry: prime per-row jet caches before the ext-coord
        // par_iter — see `warm_up_outer_caches_for_mode` doc. gam#979: only the
        // caches this `eval_mode` consumes are primed.
        if let Some(workspace) = hessian_workspace.as_ref() {
            workspace.warm_up_outer_caches_for_mode(eval_mode)?;
        }
        let (
            h_joint_unpen,
            rho_curvature_scale,
            hessian_logdet_correction,
            use_outer_curvature_derivatives,
        ) = if let Some(curvature) = family.exact_newton_outer_curvature(&inner.block_states)? {
            (
                JointHessianSource::Dense(symmetrized_square_matrix(
                    curvature.hessian,
                    total,
                    "joint exact-newton Hessian shape mismatch in joint hyper evaluator (rescaled)",
                )?),
                curvature.rho_curvature_scale,
                curvature.hessian_logdet_correction,
                true,
            )
        } else {
            let h_joint_unpen = if let Some(workspace) = hessian_workspace.as_ref() {
                exact_newton_joint_hessian_source_from_workspace(
                    workspace,
                    total,
                    MaterializationIntent::OuterEvaluation,
                    "joint exact-newton operator mismatch in joint hyper evaluator",
                )?
            } else {
                None
            };
            (
                match h_joint_unpen {
                    Some(source) => Some(source),
                    None => exact_newton_joint_hessian_symmetrized(
                        family,
                        &inner.block_states,
                        specs,
                        total,
                        "joint exact-newton Hessian shape mismatch in joint hyper evaluator",
                    )
                    .map(|source| source.map(JointHessianSource::Dense))?,
                }
                .ok_or_else(|| -> CustomFamilyError {
                    "joint exact-newton Hessian unavailable for full [rho, psi] outer calculus"
                        .to_string()
                        .into()
                })?,
                1.0,
                0.0,
                false,
            )
        };

        // Build the exact pseudologdet eigenspace for each penalty block so
        // the value, ψ gradient, ψψ Hessian, and ρψ mixed block all
        // differentiate the same log|S|_+ objective.
        let s_logdet_blocks = if include_logdet_s {
            use rayon::iter::{IntoParallelIterator, ParallelIterator};
            let block_results: Vec<Result<PenaltyPseudologdet, CustomFamilyError>> = (0..specs.len())
                .into_par_iter()
                .map(|b| {
                    let spec = &specs[b];
                    let p = spec.design.ncols();
                    let lambdas = exact_lambdas_from_log_strengths(
                        &per_block[b],
                        &format!("psi hyper logdet block {b} log strength"),
                    )?;
                    let mut s_lambda = Array2::<f64>::zeros((p, p));
                    for (k, s) in spec.penalties.iter().enumerate() {
                        s.add_scaled_to(lambdas[k], &mut s_lambda);
                    }
                    let ridge_hint = if options.ridge_policy.accounts_for_objective() {
                        for d in 0..p {
                            s_lambda[[d, d]] += ridge;
                        }
                        Some(ridge)
                    } else {
                        None
                    };
                    // No metadata-based structural-nullity hint: the
                    // PenaltyPseudologdet classifier derives the positive
                    // eigenspace from the assembled spectrum alone (issues
                    // #192/#318).
                    PenaltyPseudologdet::from_assembled(s_lambda, ridge_hint)
                        .map_err(CustomFamilyError::trial_point)
                })
                .collect();
            let blocks: Result<Vec<_>, _> = block_results.into_iter().collect();
            Some(blocks?)
        } else {
            None
        };

        let robust_jeffreys_hphi =
            custom_family_outer_jeffreys_hphi(family, &inner.block_states, specs, &ranges)?;
        let has_configured_rho_prior = !matches!(rho_prior, gam_problem::RhoPrior::Flat);
        let batched_gradient_contract_allows_override =
            batched_outer_gradient_contract_allows_override(
                robust_jeffreys_hphi
                    .as_ref()
                    .map(|(_phi, hphi, _completion)| hphi),
            );
        // The batched outer-gradient override produces the ENVELOPE gradient
        // `objective_θ + ½tr[..] − ½ld_s` only — it omits the KKT-residual
        // (one-step Newton profile) correction `−coord.gᵀq + ½qᵀ Ḣ q` that the
        // unified evaluator applies (cost-side `−½rᵀH⁻¹r`, ρ AND ψ gradient
        // derivatives) whenever the inner solve exits at β̂ with a nonzero KKT
        // residual `r = ∇_β L_pen(β̂)`. At exact KKT (`r ≈ 0`) the correction is
        // identically zero and the batched envelope gradient equals the unified
        // gradient, so the fast path is used. When the inner exit accepts a
        // non-negligible residual (near-singular blocks), the omitted term is
        // amplified by `‖H⁻¹‖·‖r‖` and the envelope gradient diverges from the
        // true derivative of the corrected objective — so fall back to the
        // unified evaluator (which carries the correction for every coordinate).
        let inner_kkt_residual_is_negligible = match inner.kkt_residual.as_ref() {
            None => true,
            Some(residual) => {
                let r = residual.as_array();
                let r_inf = r.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
                // The KKT correction's leading term `−coord.gᵀ(H⁻¹r)` is bounded
                // by `‖H⁻¹‖·‖coord.g‖·‖r‖`; treat the residual as exact only when
                // its inf-norm is at the inner solve's own KKT tolerance floor
                // (defaulting to a tight `1e-8` when the producer attached none),
                // so the fast batched path is taken on well-converged fits and
                // the unified correction path is taken whenever `r` is materially
                // nonzero.
                let tol = residual.residual_tol().unwrap_or(1.0e-8).max(1.0e-12);
                r_inf <= tol
            }
        };
        let mut batched_gradient_override: Option<Array1<f64>> = None;
        if !has_configured_rho_prior
            && batched_gradient_contract_allows_override
            && inner_kkt_residual_is_negligible
            && (eval_mode == EvalMode::ValueAndGradient
                || eval_mode == EvalMode::ValueGradientHessian)
            && let Ok(Some(batch)) = family.batched_outer_gradient_terms(
                synced_joint_states.as_ref(),
                specs,
                hyper_layout.as_ref(),
                rho_current,
                options,
                hessian_workspace.clone(),
            )
        {
            let expected = rho_dim + psi_dim;
            if batch.objective_theta.len() == expected
                && batch.trace_h_inv_hdot.len() == expected
                && batch.trace_s_pinv_sdot.len() == expected
            {
                let mut gradient = Array1::<f64>::zeros(expected);
                for j in 0..expected {
                    let trace_term = if include_logdet_h {
                        0.5 * batch.trace_h_inv_hdot[j]
                    } else {
                        0.0
                    };
                    let det_term = if include_logdet_s {
                        0.5 * batch.trace_s_pinv_sdot[j]
                    } else {
                        0.0
                    };
                    gradient[j] = batch.objective_theta[j] + trace_term - det_term;
                }
                if eval_mode == EvalMode::ValueGradientHessian {
                    batched_gradient_override = Some(gradient);
                } else {
                    let no_dh =
                        |_: &Array1<f64>| -> Result<Option<DriftDerivResult>, CustomFamilyError> { Ok(None) };
                    let no_d2h = |_: &Array1<f64>,
                                  _: &Array1<f64>|
                     -> Result<Option<DriftDerivResult>, CustomFamilyError> {
                        Ok(None)
                    };
                    let value_only = joint_outer_evaluate(
                        &inner,
                        specs,
                        &per_block,
                        rho_current,
                        &beta_flat,
                        h_joint_unpen,
                        &ranges,
                        total,
                        ridge,
                        moderidge,
                        extra_logdet_ridge,
                        rho_curvature_scale,
                        hessian_logdet_correction,
                        include_logdet_h,
                        include_logdet_s,
                        strict_spd,
                        // The batched BMS gradient contracts traces through the
                        // family's smooth pseudo-logdet operator. Pair it with the
                        // same scalar value convention; the projected-subspace
                        // value belongs only to the generic projected-gradient path.
                        false,
                        EvalMode::ValueOnly,
                        options,
                        gam_problem::RhoPrior::Flat,
                        family.pseudo_logdet_mode(),
                        &no_dh,
                        None,
                        &no_d2h,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        robust_jeffreys_hphi.clone(),
                        None,
                    )?;
                    return Ok(OuterObjectiveEvalResult {
                        objective: value_only.objective,
                        criterion_components: value_only.criterion_components,
                        gradient,
                        outer_hessian: gam_problem::HessianValue::Unavailable,
                        warm_start: value_only.warm_start,
                        inner_converged: inner.converged,
                        hyper_values: hyper_layout.values().clone(),
                        ext_mode_response_cols: None,
                        inner: inner.clone(),
                    });
                }
            }
        }

        // Build ψ HyperCoords, pair callbacks, and drift derivative callback.
        let hessian_beta_independent = !family.exact_newton_joint_hessian_beta_dependent();
        let psi_workspace = if eval_mode != EvalMode::ValueOnly
            && (eval_mode == EvalMode::ValueGradientHessian
                || family.exact_newton_joint_psi_workspace_for_first_order_terms())
        {
            family.exact_newton_joint_psi_workspace_with_options(
                synced_joint_states.as_ref(),
                specs,
                hyper_layout.as_ref(),
                options,
            )?
        } else {
            None
        };

        let rho_slice = rho_current
            .as_slice()
            .ok_or_else(|| "outer rho vector must be contiguous".to_string())?;
        let ext_bundle = if eval_mode == EvalMode::ValueOnly {
            None
        } else {
            let psi_coords = build_psi_hyper_coords(
                family,
                synced_joint_states.as_ref(),
                specs,
                hyper_layout.as_ref(),
                &beta_flat,
                rho_slice,
                penalty_counts,
                s_logdet_blocks.as_deref(),
                hessian_beta_independent,
                psi_workspace.clone(),
            )?;

            let (ext_ext_fn, rho_ext_fn, drift_fn, contracted_psi_fn) =
                if eval_mode == EvalMode::ValueGradientHessian {
                    // EXPLICIT Firth/Jeffreys ψψ VALUE second-derivative context
                    // (gam#1607). Built ONCE and shared by BOTH the per-pair
                    // `ext_ext_fn` and the contracted hook so whichever ψψ Hessian
                    // path the outer solver uses carries the `−∂²_ψΦ` term matching
                    // the gradient term `−∂_ψΦ` from `build_psi_hyper_coords`.
                    // gam#1607 / #901: the explicit-ψ Firth ψψ VALUE second
                    // derivative `−∂²_ψΦ` is the second-order analogue of the
                    // gradient term `−∂_ψΦ`. It is only well-defined when the
                    // Jeffreys information actually carries explicit ψ-dependence
                    // (`H_info ≡ H_joint`, length-scale ψ reshaping the design).
                    // For families whose Jeffreys info is the data Fisher
                    // information `XᵀWX` and whose ψ are penalty hyperparameters
                    // (design `X` fixed → `∂_ψ H_info ≡ 0`), the engine would form
                    // the second derivative from the WRONG perturbation
                    // `∂²_ψ(penalty)`; suppress the context so both the per-pair
                    // and contracted ψψ Hessian paths drop `−∂²_ψΦ` (true value 0),
                    // mirroring the gradient-side gating in `build_psi_hyper_coords`.
                    let jeffreys_ctx = if family.joint_jeffreys_information_depends_on_psi() {
                        build_jeffreys_hphi_ctx(
                            family,
                            synced_joint_states.as_ref(),
                            specs,
                            hyper_layout.as_ref(),
                            beta_flat.len(),
                        )?
                    } else {
                        None
                    };
                    let (ext_ext_fn, rho_ext_fn) = build_psi_pair_callbacks(
                        family,
                        synced_joint_states.as_ref(),
                        specs,
                        Arc::clone(&hyper_layout),
                        &beta_flat,
                        rho_slice,
                        penalty_counts,
                        s_logdet_blocks.as_deref(),
                        psi_workspace.clone(),
                        jeffreys_ctx.clone(),
                    )?;
                    // #740: build the direction-contracted ψψ hook from the same psi
                    // workspace + penalty data the per-pair `ext_ext_fn` uses, so the
                    // matrix-free outer-Hessian operator collapses the `K²` per-pair
                    // ψψ assembly to one combined-direction family row pass per
                    // matvec. `None` (no contracted family kernel) keeps the exact
                    // per-pair `ext_ext_fn` path. Built before the drift callback
                    // moves `psi_workspace`.
                    let contracted_psi_fn = build_contracted_psi_hook(
                        specs,
                        Arc::clone(&hyper_layout),
                        &beta_flat,
                        rho_slice,
                        penalty_counts,
                        s_logdet_blocks.as_deref(),
                        psi_workspace.clone(),
                        jeffreys_ctx,
                    )?;
                    let drift_fn = build_psi_drift_deriv_callback(
                        family,
                        synced_joint_states.as_ref(),
                        specs,
                        Arc::clone(&hyper_layout),
                        hessian_beta_independent,
                        psi_workspace,
                    )?;
                    (
                        Some(ext_ext_fn),
                        Some(rho_ext_fn),
                        drift_fn,
                        contracted_psi_fn,
                    )
                } else {
                    (None, None, None, None)
                };

            Some(ExtCoordBundle {
                coords: psi_coords,
                ext_ext_fn,
                rho_ext_fn,
                drift_fn,
                contracted_psi_fn,
            })
        };

        // Build derivative provider for the ρ coordinates (D_β H[v]).
        let compute_dh = exact_newton_dh_closure(
            family,
            Arc::clone(&synced_joint_states),
            specs,
            total,
            use_outer_curvature_derivatives,
            if use_outer_curvature_derivatives {
                1.0
            } else {
                rho_curvature_scale
            },
            hessian_workspace.clone(),
        );
        let compute_dh_many = if use_outer_curvature_derivatives {
            None
        } else {
            exact_newton_dh_many_closure(rho_curvature_scale, hessian_workspace.clone())
        };
        let compute_d2h = exact_newton_d2h_closure(
            family,
            Arc::clone(&synced_joint_states),
            specs,
            total,
            use_outer_curvature_derivatives,
            if use_outer_curvature_derivatives {
                1.0
            } else {
                rho_curvature_scale
            },
            hessian_workspace.clone(),
        );
        let owned_compute_dh = exact_newton_dh_closure_owned(
            family.clone(),
            Arc::clone(&synced_joint_states),
            specs.to_vec(),
            total,
            use_outer_curvature_derivatives,
            if use_outer_curvature_derivatives {
                1.0
            } else {
                rho_curvature_scale
            },
            hessian_workspace.clone(),
        );
        let owned_compute_dh_many = if use_outer_curvature_derivatives {
            None
        } else {
            exact_newton_dh_many_closure_owned(rho_curvature_scale, hessian_workspace.clone())
        };
        let owned_compute_d2h = exact_newton_d2h_closure_owned(
            family.clone(),
            Arc::clone(&synced_joint_states),
            specs.to_vec(),
            total,
            use_outer_curvature_derivatives,
            if use_outer_curvature_derivatives {
                1.0
            } else {
                rho_curvature_scale
            },
            hessian_workspace.clone(),
        );
        let compute_d2h_many = if use_outer_curvature_derivatives {
            None
        } else {
            exact_newton_d2h_many_closure(rho_curvature_scale, hessian_workspace.clone())
        };
        let owned_compute_d2h_many = if use_outer_curvature_derivatives {
            None
        } else {
            exact_newton_d2h_many_closure_owned(rho_curvature_scale, hessian_workspace.clone())
        };

        // Route through the unified path (joint_outer_evaluate → reml_laml_evaluate).
        let mut eval_result = joint_outer_evaluate(
            &inner,
            specs,
            &per_block,
            rho_current,
            &beta_flat,
            h_joint_unpen,
            &ranges,
            total,
            ridge,
            moderidge,
            extra_logdet_ridge,
            rho_curvature_scale,
            hessian_logdet_correction,
            include_logdet_h,
            include_logdet_s,
            strict_spd,
            // ψ-bearing generic path (matern/duchon marginal-slope kernel
            // length-scales): use the projected #752 generalized determinant when
            // this call owns all derivatives. If a batched first-order override
            // is pending, pair its smooth spectral gradient with the same smooth
            // pseudo-logdet scalar/Hessian convention.
            if batched_gradient_override.is_some() {
                false
            } else {
                family.use_projected_penalty_logdet()
            },
            eval_mode,
            options,
            rho_prior.clone(),
            family.pseudo_logdet_mode(),
            &compute_dh,
            compute_dh_many.as_deref(),
            &compute_d2h,
            compute_d2h_many.as_deref(),
            Some(owned_compute_dh),
            owned_compute_dh_many,
            Some(owned_compute_d2h),
            owned_compute_d2h_many,
            ext_bundle,
            None,
            custom_family_batched_outer_hessian_operator(
                family,
                synced_joint_states.as_ref(),
                specs,
                hyper_layout.as_ref(),
                rho_current,
                hessian_workspace.clone(),
                eval_mode,
            )?,
            robust_jeffreys_hphi,
            custom_family_outer_jeffreys_hphi_drift_batched(
                family,
                &inner.block_states,
                specs,
                &ranges,
            )?,
        )?;
        if let Some(gradient) = batched_gradient_override {
            eval_result.gradient = gradient;
        }
        eval_result.hyper_values = hyper_layout.values().clone();

        // The unified evaluator produces gradient/Hessian of size (rho_dim + psi_dim),
        // with ρ coordinates first and ψ coordinates appended — matching the expected
        // output order of CustomFamilyJointHyperResult.
        log::info!(
            "[STAGE] cthf_internal psi_dim={} eval_mode={:?} post_unified elapsed={:.3}s",
            psi_dim,
            eval_mode,
            cthf_internal_psi_branch_start.elapsed().as_secs_f64(),
        );
        return Ok(eval_result);
    }

    // ── ρ-only path (psi_dim == 0): route through unified evaluator ──
    //
    // Batched fast-path: if the family overrides `batched_outer_gradient_terms`,
    // factor H once at the family level and amortize all K trace computations in
    // a single streaming pass. Runs in both `ValueAndGradient` and
    // `ValueGradientHessian` modes; in VGH the Hessian still flows through the
    // standard joint_outer_evaluate path below and only the gradient is
    // replaced. See `BatchedOuterGradientTerms`. The replacement is permitted
    // only when it differentiates the same objective: if robust Jeffreys
    // curvature is nonzero, the unified H_phi-aware evaluator owns the gradient.
    let has_configured_rho_prior = !matches!(rho_prior, gam_problem::RhoPrior::Flat);
    let robust_jeffreys_hphi =
        custom_family_outer_jeffreys_hphi(family, &inner.block_states, specs, &ranges)?;
    let batched_gradient_contract_allows_override = batched_outer_gradient_contract_allows_override(
        robust_jeffreys_hphi
            .as_ref()
            .map(|(_phi, hphi, _completion)| hphi),
    );
    let mut batched_gradient_override: Option<Array1<f64>> = None;
    if !has_configured_rho_prior
        && batched_gradient_contract_allows_override
        && (eval_mode == EvalMode::ValueAndGradient || eval_mode == EvalMode::ValueGradientHessian)
    {
        let beta_flat_for_batch = flatten_state_betas(&inner.block_states, specs);
        let synced_states_for_batch = synchronized_states_from_flat_beta(
            family,
            specs,
            &inner.block_states,
            &beta_flat_for_batch,
        )?;
        let workspace_for_batch = match inner.joint_workspace.clone() {
            Some(workspace) => Some(workspace),
            None => family
                .exact_newton_joint_hessian_workspace_with_options(
                    &synced_states_for_batch,
                    specs,
                    options,
                )
                .ok()
                .flatten(),
        };
        if let Ok(Some(batch)) = family.batched_outer_gradient_terms(
            &synced_states_for_batch,
            specs,
            hyper_layout.as_ref(),
            rho_current,
            options,
            workspace_for_batch.clone(),
        ) {
            // Sanity check: batched output must match (rho_dim + psi_dim).
            let expected = rho_dim + psi_dim;
            if batch.objective_theta.len() == expected
                && batch.trace_h_inv_hdot.len() == expected
                && batch.trace_s_pinv_sdot.len() == expected
                && let Some(joint_bundle_value_only) = build_joint_hessian_closures(
                    family,
                    &inner.block_states,
                    specs,
                    total,
                    options,
                    inner.joint_workspace.clone(),
                    // The bundle's directional closures feed only the
                    // `EvalMode::ValueOnly` `joint_outer_evaluate` below — the
                    // gradient is supplied by the family's batched terms — so
                    // no directional jet cache needs priming (gam#979).
                    EvalMode::ValueOnly,
                )?
            {
                let mut gradient = Array1::<f64>::zeros(expected);
                for j in 0..expected {
                    let trace_term = if include_logdet_h {
                        0.5 * batch.trace_h_inv_hdot[j]
                    } else {
                        0.0
                    };
                    let det_term = if include_logdet_s {
                        0.5 * batch.trace_s_pinv_sdot[j]
                    } else {
                        0.0
                    };
                    gradient[j] = batch.objective_theta[j] + trace_term - det_term;
                }
                if eval_mode == EvalMode::ValueGradientHessian {
                    batched_gradient_override = Some(gradient);
                } else {
                    let JointHessianBundle {
                        source: h_joint_unpen,
                        beta_flat,
                        compute_dh,
                        compute_dh_many,
                        compute_d2h,
                        compute_d2h_many,
                        owned_compute_dh: _,
                        owned_compute_dh_many: _,
                        owned_compute_d2h: _,
                        owned_compute_d2h_many: _,
                        rho_curvature_scale,
                        hessian_logdet_correction,
                    } = joint_bundle_value_only;
                    let value_only = joint_outer_evaluate(
                        &inner,
                        specs,
                        &per_block,
                        rho_current,
                        &beta_flat,
                        h_joint_unpen,
                        &ranges,
                        total,
                        ridge,
                        moderidge,
                        extra_logdet_ridge,
                        rho_curvature_scale,
                        hessian_logdet_correction,
                        include_logdet_h,
                        include_logdet_s,
                        strict_spd,
                        // VALUE/GRADIENT CONSISTENCY: this `value_only` is paired
                        // with the family's BATCHED gradient (computed just above),
                        // which evaluates the logdet derivative through the
                        // family's `pseudo_logdet_mode` spectral operator (Smooth
                        // `r_ε` for BMS) — an internally exact antiderivative pair
                        // (value `log r_ε`, gradient `φ'=r_ε'/r_ε`). The value must
                        // therefore use the SAME spectral convention, NOT the
                        // projected #752 generalized determinant, or value and the
                        // batched gradient would describe different objectives under
                        // rank deficiency. The projected determinant is used on the
                        // non-batched path (the ψ-bearing matern marginal-slope
                        // route, gam#808/#787), where joint_outer_evaluate produces
                        // a matched projected value AND gradient in one call.
                        false,
                        EvalMode::ValueOnly,
                        options,
                        gam_problem::RhoPrior::Flat,
                        family.pseudo_logdet_mode(),
                        compute_dh.as_ref(),
                        compute_dh_many.as_deref(),
                        compute_d2h.as_ref(),
                        compute_d2h_many.as_deref(),
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        robust_jeffreys_hphi.clone(),
                        // ValueOnly: the gradient is supplied separately below, so
                        // the H_Φ mode-response drift (a gradient-only term) is not
                        // needed here.
                        None,
                    )?;
                    return Ok(OuterObjectiveEvalResult {
                        objective: value_only.objective,
                        criterion_components: value_only.criterion_components,
                        gradient,
                        outer_hessian: gam_problem::HessianValue::Unavailable,
                        warm_start: value_only.warm_start,
                        inner_converged: inner.converged,
                        hyper_values: hyper_layout.values().clone(),
                        ext_mode_response_cols: None,
                        inner: inner.clone(),
                    });
                }
            }
        }
    }

    // Try build_joint_hessian_closures which handles both exact Newton and
    // surrogate Hessian sources, then call joint_outer_evaluate with no
    // extended coordinates.
    if let Some(joint_bundle) = build_joint_hessian_closures(
        family,
        &inner.block_states,
        specs,
        total,
        options,
        inner.joint_workspace.clone(),
        // gam#979: this bundle drives the unified evaluator at the caller's
        // requested `eval_mode`, so prime exactly the directional caches that
        // mode consumes (none for value-only line-search / seed-screen probes,
        // third-only for the first-order gradient, both for the outer Hessian).
        eval_mode,
    )? {
        let JointHessianBundle {
            source: h_joint_unpen,
            beta_flat,
            compute_dh,
            compute_dh_many,
            compute_d2h,
            compute_d2h_many,
            owned_compute_dh,
            owned_compute_dh_many,
            owned_compute_d2h,
            owned_compute_d2h_many,
            rho_curvature_scale,
            hessian_logdet_correction,
        } = joint_bundle;
        let eval_result = joint_outer_evaluate(
            &inner,
            specs,
            &per_block,
            rho_current,
            &beta_flat,
            h_joint_unpen,
            &ranges,
            total,
            ridge,
            moderidge,
            extra_logdet_ridge,
            rho_curvature_scale,
            hessian_logdet_correction,
            include_logdet_h,
            include_logdet_s,
            strict_spd,
            // VALUE/GRADIENT CONSISTENCY: when a batched (Smooth-mode) gradient
            // override is pending, it will replace `eval_result.gradient` below,
            // so the value (and outer Hessian) here must use the SAME spectral
            // convention as that gradient — the family's `pseudo_logdet_mode`
            // (Smooth `r_ε`), NOT the projected #752 generalized determinant. The
            // projected determinant is used only when no batched override is
            // active (the ψ-bearing matern marginal-slope route, gam#808/#787),
            // where this call produces a matched projected value+gradient+Hessian.
            if batched_gradient_override.is_some() {
                false
            } else {
                family.use_projected_penalty_logdet()
            },
            eval_mode,
            options,
            rho_prior.clone(),
            family.pseudo_logdet_mode(),
            compute_dh.as_ref(),
            compute_dh_many.as_deref(),
            compute_d2h.as_ref(),
            compute_d2h_many.as_deref(),
            owned_compute_dh,
            owned_compute_dh_many,
            owned_compute_d2h,
            owned_compute_d2h_many,
            None, // no ext_coords when psi_dim == 0
            None,
            custom_family_batched_outer_hessian_operator(
                family,
                &inner.block_states,
                specs,
                hyper_layout.as_ref(),
                rho_current,
                inner.joint_workspace.clone(),
                eval_mode,
            )?,
            robust_jeffreys_hphi,
            custom_family_outer_jeffreys_hphi_drift_batched(
                family,
                &inner.block_states,
                specs,
                &ranges,
            )?,
        )?;

        let mut eval_result = eval_result;
        if let Some(batched_grad) = batched_gradient_override.take()
            && batched_grad.len() == eval_result.gradient.len()
        {
            eval_result.gradient = batched_grad;
        }
        eval_result.hyper_values = hyper_layout.values().clone();
        return Ok(eval_result);
    }

    // Joint Hessian unavailable via either exact Newton or surrogate.
    // The generic fallback is only mathematically defensible for single-block
    // families — multi-block families with coupled likelihood curvature require
    // the joint path.
    if family.requires_joint_outer_hyper_path() {
        return Err(
            "outer hyper-derivative evaluation requires a joint exact path for this family"
                .to_string()
                .into(),
        );
    }

    // Generic fallback: single-block only. Extract the per-block Hessian and
    // route through joint_outer_evaluate with the single block as the "joint"
    // system.
    if specs.len() != 1 {
        return Err(
            "generic outer fallback is only valid for single-block families; multi-block families must provide a joint outer path"
                .to_string()
                .into(),
        );
    }
    let eval = family.evaluate(&inner.block_states)?;
    let b = 0;
    let spec = &specs[b];
    let work = &eval.blockworking_sets[b];
    let p = spec.design.ncols();
    let mut diagonal_design = None::<DesignMatrix>;
    let h_joint_unpen = match work {
        BlockWorkingSet::Diagonal {
            working_response: _,
            working_weights,
        }
        | BlockWorkingSet::NaturalDiagonal {
            observed_curvature: working_weights,
            ..
        } => with_block_geometry(family, &inner.block_states, spec, b, |x_dyn, _| {
            let w = certify_finite_working_weights(working_weights)?;
            let (xtwx, _) = weighted_normal_equations(x_dyn, w, None)?;
            diagonal_design = Some(x_dyn.clone());
            Ok(xtwx)
        })?,
        BlockWorkingSet::ExactNewton {
            gradient: _,
            hessian,
        } => {
            if hessian.nrows() != p || hessian.ncols() != p {
                crate::bail_dim_custom!(
                    "block {b} exact-newton Hessian shape mismatch in outer gradient: got {}x{}, expected {}x{}",
                    hessian.nrows(),
                    hessian.ncols(),
                    p,
                    p
                );
            }
            hessian.to_dense()
        }
    };

    let beta_flat = inner.block_states[b].beta.clone();

    // Build a derivative provider that computes D_β H_L[direction] on demand.
    let compute_dh = |direction: &Array1<f64>| -> Result<Option<DriftDerivResult>, CustomFamilyError> {
        if !include_logdet_h {
            return Ok(None);
        }
        match work {
            BlockWorkingSet::ExactNewton { .. } => {
                match family.exact_newton_hessian_directional_derivative(
                    &inner.block_states,
                    b,
                    direction,
                )? {
                    Some(h_exact) => Ok(Some(DriftDerivResult::Dense(symmetrized_square_matrix(
                        h_exact,
                        p,
                        &format!("block {b} exact-newton dH shape mismatch"),
                    )?))),
                    None => Err(CustomFamilyError::UnsupportedConfiguration { reason: format!(
                        "missing exact-newton dH callback for block {b} while REML gradient requires H_beta term"
                    ) }),
                }
            }
            BlockWorkingSet::Diagonal {
                working_response: _,
                working_weights,
            }
            | BlockWorkingSet::NaturalDiagonal {
                observed_curvature: working_weights,
                ..
            } => {
                let x_dyn = diagonal_design.as_ref().ok_or_else(|| {
                    format!("missing dynamic design for block {b} diagonal correction")
                })?;
                let wwork = certify_finite_working_weights(working_weights)?;
                let x_dense = x_dyn.to_dense();
                let n = x_dense.nrows();

                let mut d_eta = x_dyn.matrixvectormultiply(direction);
                let geom = family.block_geometry_directional_derivative(
                    &inner.block_states,
                    b,
                    spec,
                    direction,
                )?;
                let mut correction_mat = Array2::<f64>::zeros((p, p));

                if let Some(geom_dir) = geom {
                    d_eta += &geom_dir.d_offset;
                    if let Some(dx) = geom_dir.d_design {
                        d_eta += &dx.dot(&beta_flat);
                        let mut wx = x_dense.clone();
                        let mut wdx = dx.clone();
                        ndarray::Zip::from(wx.rows_mut())
                            .and(wdx.rows_mut())
                            .and(wwork.view())
                            .par_for_each(|mut wxr, mut wdxr, &wi| {
                                if wi != 1.0 {
                                    wxr.mapv_inplace(|v| v * wi);
                                    wdxr.mapv_inplace(|v| v * wi);
                                }
                            });
                        // Same X'(W·Y) pattern as the parallel sibling at
                        // line ~9258; route through faer for SIMD GEMM
                        // (n × p² flops at large-scale moderate scale).
                        correction_mat += &fast_atb(&dx, &wx);
                        correction_mat += &fast_atb(&x_dense, &wdx);
                    }
                }

                let dw = family
                    .diagonalworking_weights_directional_derivative(
                        &inner.block_states,
                        b,
                        &d_eta,
                    )?
                    .ok_or_else(|| {
                        format!(
                            "missing diagonal dW callback for block {b} while REML gradient requires H_beta term"
                        )
                    })?;
                if dw.len() != n {
                    return Err(CustomFamilyError::DimensionMismatch {
                        reason: format!(
                            "block {b} diagonal dW length mismatch: got {}, expected {}",
                            dw.len(),
                            n
                        ),
                    });
                }
                let mut scaled_x = x_dense.clone();
                ndarray::Zip::from(scaled_x.rows_mut())
                    .and(&dw)
                    .par_for_each(|mut sr, &dwi| sr.mapv_inplace(|v| v * dwi));
                // X'(diag(dW)·X) outer correction term — faer route, same
                // rationale as above.
                correction_mat += &fast_atb(&x_dense, &scaled_x);

                Ok(Some(DriftDerivResult::Dense(correction_mat)))
            }
        }
    };

    // Build a derivative provider that computes D²_β H_L[u, v] on demand.
    let compute_d2h = |u: &Array1<f64>,
                       v: &Array1<f64>|
     -> Result<Option<DriftDerivResult>, CustomFamilyError> {
        if !include_logdet_h {
            return Ok(None);
        }
        match work {
            BlockWorkingSet::ExactNewton { .. } => {
                match family.exact_newton_hessian_second_directional_derivative(
                    &inner.block_states,
                    b,
                    u,
                    v,
                )? {
                    Some(h_exact) => Ok(Some(DriftDerivResult::Dense(symmetrized_square_matrix(
                        h_exact,
                        p,
                        &format!("block {b} exact-newton d2H shape mismatch"),
                    )?))),
                    None => Err(CustomFamilyError::UnsupportedConfiguration { reason: format!(
                        "missing exact-newton d2H callback for block {b} while REML Hessian requires H_beta_beta term"
                    ) }),
                }
            }
            BlockWorkingSet::Diagonal {
                working_response: _,
                working_weights: _,
            }
            | BlockWorkingSet::NaturalDiagonal { .. } => {
                let x_dyn = diagonal_design.as_ref().ok_or_else(|| {
                    format!("missing dynamic design for block {b} diagonal second correction")
                })?;
                let x_dense = x_dyn.to_dense();
                let n = x_dense.nrows();

                let reject_second_order_geometry = |label: &str,
                                                    geom: Option<
                    BlockGeometryDirectionalDerivative,
                >|
                 -> Result<(), CustomFamilyError> {
                    if let Some(geom_dir) = geom {
                        let has_offset = geom_dir.d_offset.iter().any(|value| *value != 0.0);
                        if geom_dir.d_design.is_some() || has_offset {
                            return Err(CustomFamilyError::UnsupportedConfiguration { reason: format!(
                                "block {b} diagonal d2H requires second-order block-geometry derivatives for {label}; use an exact-newton or joint outer path"
                            ) });
                        }
                    }
                    Ok(())
                };
                reject_second_order_geometry(
                    "first direction",
                    family.block_geometry_directional_derivative(
                        &inner.block_states,
                        b,
                        spec,
                        u,
                    )?,
                )?;
                reject_second_order_geometry(
                    "second direction",
                    family.block_geometry_directional_derivative(
                        &inner.block_states,
                        b,
                        spec,
                        v,
                    )?,
                )?;

                let d_eta_u = x_dyn.matrixvectormultiply(u);
                let d_eta_v = x_dyn.matrixvectormultiply(v);
                let d2w = family
                    .diagonalworking_weights_second_directional_derivative(
                        &inner.block_states,
                        b,
                        &d_eta_u,
                        &d_eta_v,
                    )?
                    .ok_or_else(|| {
                        format!(
                            "missing diagonal d2W callback for block {b} while REML Hessian requires H_beta_beta term"
                        )
                    })?;
                if d2w.len() != n {
                    return Err(CustomFamilyError::DimensionMismatch {
                        reason: format!(
                            "block {b} diagonal d2W length mismatch: got {}, expected {}",
                            d2w.len(),
                            n
                        ),
                    });
                }
                let mut scaled_x = x_dense.clone();
                ndarray::Zip::from(scaled_x.rows_mut())
                    .and(&d2w)
                    .par_for_each(|mut sr, &d2wi| sr.mapv_inplace(|value| value * d2wi));
                Ok(Some(DriftDerivResult::Dense(fast_atb(&x_dense, &scaled_x))))
            }
        }
    };

    let mut eval_result = joint_outer_evaluate(
        &inner,
        specs,
        &per_block,
        rho_current,
        &beta_flat,
        JointHessianSource::Dense(h_joint_unpen),
        &ranges,
        total,
        ridge,
        moderidge,
        extra_logdet_ridge,
        1.0,
        0.0,
        include_logdet_h,
        include_logdet_s,
        strict_spd,
        family.use_projected_penalty_logdet(),
        eval_mode,
        options,
        rho_prior,
        family.pseudo_logdet_mode(),
        &compute_dh,
        None,
        &compute_d2h,
        None,
        None,
        None,
        None,
        None,
        None, // no ext_coords for generic single-block fallback
        None,
        custom_family_batched_outer_hessian_operator(
            family,
            &inner.block_states,
            specs,
            hyper_layout.as_ref(),
            rho_current,
            inner.joint_workspace.clone(),
            eval_mode,
        )?,
        robust_jeffreys_hphi,
        custom_family_outer_jeffreys_hphi_drift_batched(
            family,
            &inner.block_states,
            specs,
            &ranges,
        )?,
    )?;

    eval_result.hyper_values = hyper_layout.values().clone();
    Ok(eval_result)
}

pub fn evaluate_custom_family_joint_hyper<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
    rho_current: &Array1<f64>,
    hyper_layout: &CustomFamilyHyperLayout,
    warm_start: Option<&CustomFamilyWarmStart>,
    eval_mode: EvalMode,
) -> Result<CustomFamilyJointHyperResult, CustomFamilyError> {
    Ok(evaluate_custom_family_joint_hyper_owned(
        family,
        specs,
        options,
        rho_current,
        hyper_layout,
        warm_start,
        eval_mode,
    )?
    .result)
}

/// Evaluate a joint hyperparameter point and retain the exact coefficient mode
/// that produced its objective and derivative payload.
pub fn evaluate_custom_family_joint_hyper_owned<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
    rho_current: &Array1<f64>,
    hyper_layout: &CustomFamilyHyperLayout,
    warm_start: Option<&CustomFamilyWarmStart>,
    eval_mode: EvalMode,
) -> Result<CustomFamilyJointHyperOwnedResult, CustomFamilyError> {
    let penalty_counts = validate_blockspecs(specs)?;
    let has_psi_derivatives = !hyper_layout.is_empty();
    let (eval_options, strict_warm_start) =
        derivative_quality_options_and_warm_start(options, warm_start, has_psi_derivatives);
    let eval_result = evaluate_custom_family_hyper_internal(
        family,
        specs,
        &eval_options,
        &penalty_counts,
        rho_current,
        hyper_layout,
        strict_warm_start
            .as_ref()
            .map(|w| &w.inner)
            .or_else(|| warm_start.map(|w| &w.inner)),
        gam_problem::RhoPrior::Flat,
        eval_mode,
    )?;
    Ok(outer_eval_result_into_joint_hyper_owned_result(eval_result))
}


pub struct CustomFamilyJointHyperModeSelection {
    pub result: CustomFamilyJointHyperResult,
    pub selected_candidate: usize,
    pub screened_objectives: Vec<Option<f64>>,
    pub rejected_candidates: Vec<Option<String>>,
    /// Exact owned coefficient mode that produced `result`.
    ///
    /// Crate-internal finalization consumes this directly so a fixed-hyper fit
    /// cannot re-enter the nonconvex inner solver and silently change basins.
    pub(crate) mode: CustomFamilyOwnedMode,
}

/// Preserve the scalar value that ranked a coefficient-mode candidate while
/// proving that derivative assembly still describes that same objective.
///
/// Value-only and derivative-bearing paths can legitimately use different
/// reduction trees, so bitwise equality is not a valid identity test. The
/// outer optimizer already owns the canonical roundoff envelope for two
/// independently assembled values at the same point; mode selection uses that
/// same contract and retains the screened value as the scalar authority.
fn canonicalize_screened_objective(
    screened: f64,
    derivative_sample: f64,
    selected_candidate: usize,
) -> Result<f64, CustomFamilyError> {
    let bound = gam_solve::rho_optimizer::outer_value_agreement_bound(screened, derivative_sample);
    let disagreement = (screened - derivative_sample).abs();
    if !screened.is_finite()
        || !derivative_sample.is_finite()
        || !disagreement.is_finite()
        || disagreement > bound
    {
        return Err(CustomFamilyError::UnsupportedConfiguration {
            reason: format!(
                "best coefficient-mode candidate {selected_candidate} changed profile objective \
                 between value screening and derivative assembly: screened={screened:.16e}, \
                 derivative={derivative_sample:.16e}, disagreement={disagreement:.3e}, \
                 roundoff bound={bound:.3e}"
            ),
        });
    }
    Ok(screened)
}

/// Profile a nonconvex coefficient mode without assembling expensive outer
/// derivatives for every candidate.
///
/// Every candidate is solved once at the requested derivative quality while
/// assembling only its value. The finite objective winner (candidate order
/// breaks exact ties) owns the exact [`BlockwiseInnerResult`] used for that
/// value; requested derivatives are assembled directly from that same mode.
/// If the winning branch cannot provide the requested derivative payload, the
/// evaluation errors instead of silently changing the profiled objective by
/// selecting a worse coefficient basin.
pub fn evaluate_custom_family_joint_hyper_best_mode_shared<
    F: CustomFamily + Clone + Send + Sync + 'static,
>(
    family: &F,
    specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
    rho_current: &Array1<f64>,
    hyper_layout: SharedCustomFamilyHyperLayout,
    candidates: &[Option<CustomFamilyWarmStart>],
    eval_mode: EvalMode,
) -> Result<CustomFamilyJointHyperModeSelection, CustomFamilyError> {
    if candidates.is_empty() {
        return Err(CustomFamilyError::InvalidInput {
            context: "evaluate_custom_family_joint_hyper_best_mode_shared",
            reason: "at least one coefficient-mode candidate is required".to_string(),
        });
    }

    let mut screened_objectives = vec![None; candidates.len()];
    let mut rejected_candidates = vec![None; candidates.len()];
    let mut screened_results: Vec<Option<OuterObjectiveEvalResult>> =
        (0..candidates.len()).map(|_| None).collect();
    let penalty_counts = validate_blockspecs(specs)?;
    let has_psi_derivatives = !hyper_layout.is_empty();
    for (candidate_idx, warm_start) in candidates.iter().enumerate() {
        let (eval_options, strict_warm_start) = derivative_quality_options_and_warm_start(
            options,
            warm_start.as_ref(),
            has_psi_derivatives,
        );
        let candidate = match evaluate_custom_family_hyper_internal_shared(
            family,
            specs,
            &eval_options,
            &penalty_counts,
            rho_current,
            Arc::clone(&hyper_layout),
            strict_warm_start
                .as_ref()
                .map(|warm| &warm.inner)
                .or_else(|| warm_start.as_ref().map(|warm| &warm.inner)),
            gam_problem::RhoPrior::Flat,
            EvalMode::ValueOnly,
            eval_mode,
            None,
        ) {
            Ok(candidate) => candidate,
            Err(error) => {
                rejected_candidates[candidate_idx] = Some(format!("evaluator error: {error}"));
                continue;
            }
        };
        if !candidate.inner_converged {
            rejected_candidates[candidate_idx] =
                Some("inner coefficient solve did not converge".to_string());
            continue;
        }
        if !candidate.objective.is_finite() {
            rejected_candidates[candidate_idx] =
                Some("profile objective was non-finite".to_string());
            continue;
        }
        screened_objectives[candidate_idx] = Some(candidate.objective);
        screened_results[candidate_idx] = Some(candidate);
    }

    let mut ranked_candidates: Vec<usize> = screened_objectives
        .iter()
        .enumerate()
        .filter_map(|(idx, objective)| objective.map(|_| idx))
        .collect();
    ranked_candidates.sort_by(|left, right| {
        screened_objectives[*left]
            .expect("ranked candidate has a finite objective")
            .total_cmp(
                &screened_objectives[*right].expect("ranked candidate has a finite objective"),
            )
            .then_with(|| left.cmp(right))
    });
    if ranked_candidates.is_empty() {
        let reasons = rejected_candidates
            .iter()
            .enumerate()
            .map(|(idx, reason)| {
                format!(
                    "candidate {idx}: {}",
                    reason.as_deref().unwrap_or("no finite converged result")
                )
            })
            .collect::<Vec<_>>()
            .join("; ");
        return Err(CustomFamilyError::UnsupportedConfiguration {
            reason: format!(
                "no coefficient-mode candidate produced a finite converged profile objective: {reasons}"
            ),
        });
    }

    if matches!(eval_mode, EvalMode::ValueOnly) {
        let selected_candidate = ranked_candidates[0];
        let owned = outer_eval_result_into_joint_hyper_owned_result(
            screened_results[selected_candidate]
                .take()
                .expect("ranked candidate retains its screened result"),
        );
        return Ok(CustomFamilyJointHyperModeSelection {
            result: owned.result,
            selected_candidate,
            screened_objectives,
            rejected_candidates,
            mode: owned.mode,
        });
    }

    let selected_candidate = ranked_candidates[0];
    let screened_winner = screened_results[selected_candidate]
        .take()
        .expect("ranked candidate retains its screened result");
    let screened_objective = screened_winner.objective;
    let selected_inner = screened_winner.inner;
    let (eval_options, _) =
        derivative_quality_options_and_warm_start(options, None, has_psi_derivatives);
    let mut derivative_eval = evaluate_custom_family_hyper_internal_shared(
        family,
        specs,
        &eval_options,
        &penalty_counts,
        rho_current,
        Arc::clone(&hyper_layout),
        None,
        gam_problem::RhoPrior::Flat,
        eval_mode,
        eval_mode,
        Some(selected_inner),
    )
    .map_err(|error| CustomFamilyError::UnsupportedConfiguration {
        reason: format!(
            "best coefficient-mode candidate {selected_candidate} failed requested derivative assembly: {error}"
        ),
    })?;
    let derivative_objective = derivative_eval.objective;
    derivative_eval.objective = canonicalize_screened_objective(
        screened_objective,
        derivative_objective,
        selected_candidate,
    )?;
    derivative_eval.criterion_components[0] += derivative_eval.objective - derivative_objective;
    validate_requested_best_mode_derivatives(
        &derivative_eval,
        eval_mode,
        rho_current.len() + hyper_layout.len(),
        selected_candidate,
    )?;
    let owned = outer_eval_result_into_joint_hyper_owned_result(derivative_eval);
    Ok(CustomFamilyJointHyperModeSelection {
        result: owned.result,
        selected_candidate,
        screened_objectives,
        rejected_candidates,
        mode: owned.mode,
    })
}


/// Upgrade a value-only coefficient-mode selection at the identical hyperpoint
/// into its requested analytic derivative payload without re-entering the
/// nonconvex inner solver.
///
/// Ownership is the identity proof: the supplied selection carries the exact
/// converged `BlockwiseInnerResult` that produced its screened scalar value.
/// Smoothing and family-hyper coordinates must match bit-for-bit; a warm start
/// or numerically-near point is deliberately insufficient.
pub fn upgrade_custom_family_joint_hyper_mode_shared<
    F: CustomFamily + Clone + Send + Sync + 'static,
>(
    family: &F,
    specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
    rho_current: &Array1<f64>,
    hyper_layout: SharedCustomFamilyHyperLayout,
    selection: CustomFamilyJointHyperModeSelection,
    eval_mode: EvalMode,
) -> Result<CustomFamilyJointHyperModeSelection, CustomFamilyError> {
    if matches!(eval_mode, EvalMode::ValueOnly) {
        return Ok(selection);
    }

    let CustomFamilyJointHyperModeSelection {
        result: screened_result,
        selected_candidate,
        screened_objectives,
        rejected_candidates,
        mode,
    } = selection;
    let CustomFamilyOwnedMode {
        objective: owned_objective,
        rho: owned_rho,
        hyper_values: owned_hyper_values,
        inner: selected_inner,
    } = mode;

    let same_rho = owned_rho.len() == rho_current.len()
        && owned_rho
            .iter()
            .zip(rho_current.iter())
            .all(|(owned, current)| owned.to_bits() == current.to_bits());
    let current_hyper_values = hyper_layout.values();
    let same_hyper_values = owned_hyper_values.len() == current_hyper_values.len()
        && owned_hyper_values
            .iter()
            .zip(current_hyper_values.iter())
            .all(|(owned, current)| owned.to_bits() == current.to_bits());
    if !same_rho || !same_hyper_values {
        return Err(CustomFamilyError::InvalidInput {
            context: "upgrade_custom_family_joint_hyper_mode_shared",
            reason: format!(
                "owned coefficient mode belongs to a different hyperpoint:                  rho_match={same_rho}, family_hyper_match={same_hyper_values}"
            ),
        });
    }
    if owned_objective.to_bits() != screened_result.objective.to_bits() {
        return Err(CustomFamilyError::InvalidInput {
            context: "upgrade_custom_family_joint_hyper_mode_shared",
            reason: format!(
                "owned coefficient mode objective {:.16e} does not match screened authority {:.16e}",
                owned_objective, screened_result.objective,
            ),
        });
    }

    let penalty_counts = validate_blockspecs(specs)?;
    let has_psi_derivatives = !hyper_layout.is_empty();
    let (eval_options, _) =
        derivative_quality_options_and_warm_start(options, None, has_psi_derivatives);
    let mut derivative_eval = evaluate_custom_family_hyper_internal_shared(
        family,
        specs,
        &eval_options,
        &penalty_counts,
        rho_current,
        Arc::clone(&hyper_layout),
        None,
        gam_problem::RhoPrior::Flat,
        eval_mode,
        eval_mode,
        Some(selected_inner),
    )
    .map_err(|error| CustomFamilyError::UnsupportedConfiguration {
        reason: format!(
            "owned coefficient mode failed requested derivative assembly: {error}"
        ),
    })?;
    let derivative_objective = derivative_eval.objective;
    derivative_eval.objective = canonicalize_screened_objective(
        screened_result.objective,
        derivative_objective,
        selected_candidate,
    )?;
    derivative_eval.criterion_components[0] +=
        derivative_eval.objective - derivative_objective;
    validate_requested_best_mode_derivatives(
        &derivative_eval,
        eval_mode,
        rho_current.len() + hyper_layout.len(),
        selected_candidate,
    )?;
    let owned = outer_eval_result_into_joint_hyper_owned_result(derivative_eval);
    Ok(CustomFamilyJointHyperModeSelection {
        result: owned.result,
        selected_candidate,
        screened_objectives,
        rejected_candidates,
        mode: owned.mode,
    })
}

#[cfg(test)]
mod mode_selection_value_tests {
    use super::*;

    #[test]
    fn screened_objective_is_canonical_within_shared_roundoff_envelope() {
        let screened = 100.0;
        let bound = gam_solve::rho_optimizer::outer_value_agreement_bound(screened, screened);
        let derivative_sample = screened + 0.25 * bound;
        let canonical =
            canonicalize_screened_objective(screened, derivative_sample, 0).expect("same value");
        assert_eq!(canonical.to_bits(), screened.to_bits());

        let inconsistent_sample = screened + 2.0 * bound;
        assert!(
            canonicalize_screened_objective(screened, inconsistent_sample, 0).is_err(),
            "a derivative lane outside the shared roundoff envelope must be rejected"
        );
    }
}

fn validate_requested_best_mode_derivatives(
    result: &OuterObjectiveEvalResult,
    eval_mode: EvalMode,
    expected_theta_dim: usize,
    selected_candidate: usize,
) -> Result<(), CustomFamilyError> {
    if !result.inner_converged
        || !result.objective.is_finite()
        || result.gradient.len() != expected_theta_dim
        || result.gradient.iter().any(|value| !value.is_finite())
    {
        return Err(CustomFamilyError::UnsupportedConfiguration {
            reason: format!(
                "best coefficient-mode candidate {selected_candidate} did not produce finite, converged requested derivatives of dimension {expected_theta_dim}"
            ),
        });
    }
    if eval_mode != EvalMode::ValueGradientHessian {
        return Ok(());
    }
    if !result.outer_hessian.is_analytic() || result.outer_hessian.dim() != Some(expected_theta_dim)
    {
        return Err(CustomFamilyError::UnsupportedConfiguration {
            reason: format!(
                "best coefficient-mode candidate {selected_candidate} did not produce an analytic {expected_theta_dim}x{expected_theta_dim} Hessian"
            ),
        });
    }
    let dense = result
        .outer_hessian
        .materialize_dense()
        .map_err(|error| CustomFamilyError::UnsupportedConfiguration {
            reason: format!(
                "best coefficient-mode candidate {selected_candidate} Hessian materialization failed: {error}"
            ),
        })?
        .ok_or_else(|| CustomFamilyError::UnsupportedConfiguration {
            reason: format!(
                "best coefficient-mode candidate {selected_candidate} did not expose an analytic Hessian"
            ),
        })?;
    if dense.dim() != (expected_theta_dim, expected_theta_dim)
        || dense.iter().any(|value| !value.is_finite())
    {
        return Err(CustomFamilyError::UnsupportedConfiguration {
            reason: format!(
                "best coefficient-mode candidate {selected_candidate} materialized Hessian was not finite with shape {expected_theta_dim}x{expected_theta_dim}"
            ),
        });
    }
    Ok(())
}

pub(crate) fn derivative_quality_options_and_warm_start(
    options: &BlockwiseFitOptions,
    warm_start: Option<&CustomFamilyWarmStart>,
    has_psi_derivatives: bool,
) -> (BlockwiseFitOptions, Option<CustomFamilyWarmStart>) {
    const DIRECT_JOINT_HYPER_MIN_CYCLES: usize = 200;

    let mut eval_options = options.clone();
    // With zero ψ coordinates this API is the rho-only outer surface. Preserve
    // its coefficient-solve contract exactly.
    if !has_psi_derivatives {
        return (eval_options, None);
    }

    // A profiled ψ derivative differentiates F_β(β̂, θ) = 0. Its coefficient
    // mode therefore cannot be certified LESS accurately than the caller asked:
    // replacing a tight inner tolerance with a looser outer tolerance changes
    // β̂(θ), while the IFT still treats the selected mode as stationary. That
    // silently made the analytic response disagree with finite differences on
    // the selected Matérn profile (#2460).
    //
    // The relation is one-way. A stricter outer target may require a tighter
    // coefficient solve, but a looser outer target grants no authority to relax
    // the inner stationarity equation. No scale floor or size-dependent escape
    // belongs here: if the requested inner contract cannot be reached, the inner
    // solve must report that honestly instead of returning derivatives of a
    // different profiled objective.
    let derivative_inner_tol = eval_options.inner_tol.min(eval_options.outer_tol);
    let tighten =
        eval_options.inner_max_cycles > 1 && derivative_inner_tol < eval_options.inner_tol;
    let psi_safe_warm_start = warm_start_without_cached_inner_for_psi_derivatives(
        warm_start.map(|warm| &warm.inner),
        true,
    )
    .map(|inner| CustomFamilyWarmStart { inner });
    // A BUDGET IS NOT A TOLERANCE, and gating one on the other is why this
    // floor never fired.
    //
    // `tighten` asks whether the OUTER tolerance is stricter than the inner one.
    // Every model-level constructor sets `inner_tol` and `outer_tol` from the
    // same user scalar, so `derivative_inner_tol == inner_tol`, `tighten` is
    // false, and the cycle floor below was unreachable on exactly the paths that
    // need it. Those same constructors also source `inner_max_cycles` from the
    // user's OUTER `max_iter` -- typically 40, against this crate's own
    // `DEFAULT_CUSTOM_FAMILY_INNER_MAX_CYCLES = 1200`. A joint Newton solving a
    // coupled multi-block Hessian for an analytic outer gradient was therefore
    // cut off at a few percent of its intended budget and reported
    //
    //   custom-family inner solve did not converge after 40 cycle(s)
    //
    // which the outer search then had to refuse. The refusal is correct; the
    // starvation upstream of it is the defect.
    //
    // The two knobs are now separated. The cycle floor applies whenever this
    // evaluation carries psi derivatives, because that is what makes the solve
    // expensive. The TOLERANCE change keeps its original one-way rule: a
    // stricter outer target may demand a tighter inner solve, but a looser one
    // grants no authority to relax the inner stationarity equation (#2460).
    // `inner_max_cycles > 1` is preserved so a deliberate single-cycle probe is
    // still honoured.
    if eval_options.inner_max_cycles > 1 {
        eval_options.inner_max_cycles = eval_options
            .inner_max_cycles
            .max(DIRECT_JOINT_HYPER_MIN_CYCLES);
    }
    if !tighten {
        return (eval_options, psi_safe_warm_start);
    }
    eval_options.inner_tol = derivative_inner_tol;
    (eval_options, psi_safe_warm_start)
}

pub fn joint_hyper_options_for_outer_tolerance(
    options: &BlockwiseFitOptions,
    outer_tol: f64,
) -> BlockwiseFitOptions {
    let mut eval_options = options.clone();
    eval_options.outer_tol = eval_options.outer_tol.max(outer_tol);
    eval_options
}

pub(crate) fn evaluate_custom_family_joint_hyper_efs_internal_shared<
    F: CustomFamily + Clone + Send + Sync + 'static,
>(
    family: &F,
    specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
    penalty_counts: &[usize],
    rho_current: &Array1<f64>,
    hyper_layout: SharedCustomFamilyHyperLayout,
    warm_start: Option<&ConstrainedWarmStart>,
) -> Result<
    (
        gam_problem::EfsEval,
        ConstrainedWarmStart,
        bool,
        BlockwiseInnerResult,
    ),
    CustomFamilyError,
> {
    if hyper_layout.block_count() != specs.len() {
        crate::bail_dim_custom!(
            "joint hyper layout block count mismatch: got {}, expected {}",
            hyper_layout.block_count(),
            specs.len()
        );
    }
    if penalty_counts.len() != specs.len() {
        crate::bail_dim_custom!(
            "joint hyper penalty-count block mismatch: got {}, expected {}",
            penalty_counts.len(),
            specs.len()
        );
    }

    let rho_dim = penalty_counts.iter().sum::<usize>();
    let psi_dim = hyper_layout.len();
    if psi_dim == 0 {
        return Err(CustomFamilyError::InvalidInput {
            context: "evaluate_custom_family_joint_hyper_efs",
            reason: "joint hyper EFS requires at least one ψ coordinate".to_string(),
        });
    }
    if rho_current.len() != rho_dim {
        crate::bail_dim_custom!(
            "joint hyper rho dimension mismatch: got {}, expected {} (psi={})",
            rho_current.len(),
            rho_dim,
            psi_dim
        );
    }

    let include_logdet_h = include_exact_newton_logdet_h(family, options);
    let include_logdet_s = include_exact_newton_logdet_s(family, options);
    let strict_spd = use_exact_newton_strict_spd(family);
    let per_block = split_log_lambdas(rho_current, penalty_counts)?;
    let psi_safe_warm_start = warm_start_without_cached_inner_for_psi_derivatives(warm_start, true);
    let mut inner = inner_blockwise_fit(
        family,
        specs,
        &per_block,
        options,
        psi_safe_warm_start.as_ref().or(warm_start),
    )?;
    if !inner.converged {
        let theta_dim = rho_dim + psi_dim;
        log::warn!(
            "[OUTER] custom-family joint-hyper EFS inner solve did not converge after {} cycle(s); \
             skipping joint-hyper EFS derivative assembly for theta_dim={} (rho_dim={}, psi_dim={})",
            inner.cycles,
            theta_dim,
            rho_dim,
            psi_dim,
        );
        let (eval, warm, converged) = nonconverged_outer_efs_result(
            &inner,
            rho_current,
            theta_dim,
            "custom-family joint-hyper EFS non-converged inner solve",
        )
        .map_err(CustomFamilyError::from)?;
        return Ok((eval, warm, converged, inner));
    }
    let ridge = effective_solverridge(options.ridge_floor);
    let moderidge = if options.ridge_policy.accounts_for_objective() {
        ridge
    } else {
        0.0
    };
    let extra_logdet_ridge = 0.0;

    refresh_all_block_etas(family, specs, &mut inner.block_states)?;
    let ranges = block_param_ranges(specs);
    let total = ranges.last().map(|(_, e)| *e).unwrap_or(0);

    let beta_flat = flatten_state_betas(&inner.block_states, specs);
    let synced_joint_states = Arc::new(synchronized_states_from_flat_beta(
        family,
        specs,
        &inner.block_states,
        &beta_flat,
    )?);
    let hessian_workspace = family.exact_newton_joint_hessian_workspace_with_options(
        synced_joint_states.as_ref(),
        specs,
        options,
    )?;
    // Outer-eval entry: prime per-row jet caches before the ext-coord
    // par_iter — see `warm_up_outer_caches_for_mode` doc. The EFS evaluator
    // always assembles the first-order fixed-point gradient terms, so it
    // consumes the third-derivative directional cache (gam#979).
    if let Some(workspace) = hessian_workspace.as_ref() {
        workspace.warm_up_outer_caches_for_mode(EvalMode::ValueAndGradient)?;
    }
    let (
        h_joint_unpen,
        rho_curvature_scale,
        hessian_logdet_correction,
        use_outer_curvature_derivatives,
    ) = if let Some(curvature) = family.exact_newton_outer_curvature(&inner.block_states)? {
        (
            JointHessianSource::Dense(symmetrized_square_matrix(
                curvature.hessian,
                total,
                "joint exact-newton Hessian shape mismatch in joint hyper EFS evaluator (rescaled)",
            )?),
            curvature.rho_curvature_scale,
            curvature.hessian_logdet_correction,
            true,
        )
    } else {
        let h_joint_unpen = if let Some(workspace) = hessian_workspace.as_ref() {
            exact_newton_joint_hessian_source_from_workspace(
                workspace,
                total,
                MaterializationIntent::OuterEvaluation,
                "joint exact-newton operator mismatch in joint hyper EFS evaluator",
            )?
        } else {
            None
        };
        (
            match h_joint_unpen {
                Some(source) => Some(source),
                None => exact_newton_joint_hessian_symmetrized(
                    family,
                    &inner.block_states,
                    specs,
                    total,
                    "joint exact-newton Hessian shape mismatch in joint hyper EFS evaluator",
                )
                .map(|source| source.map(JointHessianSource::Dense))?,
            }
            .ok_or_else(|| -> CustomFamilyError {
                "joint exact-newton Hessian unavailable for full [rho, psi] fixed-point outer calculus"
                    .to_string()
                    .into()
            })?,
            1.0,
            0.0,
            false,
        )
    };

    let s_logdet_blocks = if include_logdet_s {
        use rayon::iter::{IntoParallelIterator, ParallelIterator};
        let block_results: Vec<Result<PenaltyPseudologdet, CustomFamilyError>> = (0..specs.len())
            .into_par_iter()
            .map(|b| {
                let spec = &specs[b];
                let p = spec.design.ncols();
                let lambdas = exact_lambdas_from_log_strengths(
                    &per_block[b],
                    &format!("psi fixed-point logdet block {b} log strength"),
                )?;
                let mut s_lambda = Array2::<f64>::zeros((p, p));
                for (k, s) in spec.penalties.iter().enumerate() {
                    s.add_scaled_to(lambdas[k], &mut s_lambda);
                }
                let ridge_hint = if options.ridge_policy.accounts_for_objective() {
                    for d in 0..p {
                        s_lambda[[d, d]] += ridge;
                    }
                    Some(ridge)
                } else {
                    None
                };
                // No metadata-based structural-nullity hint: the
                // PenaltyPseudologdet classifier derives the positive
                // eigenspace from the assembled spectrum alone (issues
                // #192/#318).
                PenaltyPseudologdet::from_assembled(s_lambda, ridge_hint)
                    .map_err(CustomFamilyError::trial_point)
            })
            .collect();
        let blocks: Result<Vec<_>, _> = block_results.into_iter().collect();
        Some(blocks?)
    } else {
        None
    };

    let hessian_beta_independent = !family.exact_newton_joint_hessian_beta_dependent();
    let psi_workspace = if family.exact_newton_joint_psi_workspace_for_first_order_terms() {
        family.exact_newton_joint_psi_workspace_with_options(
            synced_joint_states.as_ref(),
            specs,
            hyper_layout.as_ref(),
            options,
        )?
    } else {
        None
    };
    let rho_slice = rho_current
        .as_slice()
        .ok_or_else(|| "outer rho vector must be contiguous".to_string())?;
    let psi_coords = build_psi_hyper_coords(
        family,
        synced_joint_states.as_ref(),
        specs,
        hyper_layout.as_ref(),
        &beta_flat,
        rho_slice,
        penalty_counts,
        s_logdet_blocks.as_deref(),
        hessian_beta_independent,
        psi_workspace.clone(),
    )?;
    let ext_bundle = ExtCoordBundle {
        coords: psi_coords,
        ext_ext_fn: None,
        rho_ext_fn: None,
        drift_fn: None,
        contracted_psi_fn: None,
    };

    let compute_dh = exact_newton_dh_closure(
        family,
        Arc::clone(&synced_joint_states),
        specs,
        total,
        use_outer_curvature_derivatives,
        if use_outer_curvature_derivatives {
            1.0
        } else {
            rho_curvature_scale
        },
        hessian_workspace.clone(),
    );
    let compute_dh_many = if use_outer_curvature_derivatives {
        None
    } else {
        exact_newton_dh_many_closure(rho_curvature_scale, hessian_workspace.clone())
    };
    let compute_d2h = exact_newton_d2h_closure(
        family,
        Arc::clone(&synced_joint_states),
        specs,
        total,
        use_outer_curvature_derivatives,
        if use_outer_curvature_derivatives {
            1.0
        } else {
            rho_curvature_scale
        },
        hessian_workspace.clone(),
    );
    let owned_compute_dh = exact_newton_dh_closure_owned(
        family.clone(),
        Arc::clone(&synced_joint_states),
        specs.to_vec(),
        total,
        use_outer_curvature_derivatives,
        if use_outer_curvature_derivatives {
            1.0
        } else {
            rho_curvature_scale
        },
        hessian_workspace.clone(),
    );
    let owned_compute_dh_many = if use_outer_curvature_derivatives {
        None
    } else {
        exact_newton_dh_many_closure_owned(rho_curvature_scale, hessian_workspace.clone())
    };
    let owned_compute_d2h = exact_newton_d2h_closure_owned(
        family.clone(),
        Arc::clone(&synced_joint_states),
        specs.to_vec(),
        total,
        use_outer_curvature_derivatives,
        if use_outer_curvature_derivatives {
            1.0
        } else {
            rho_curvature_scale
        },
        hessian_workspace.clone(),
    );
    let compute_d2h_many = if use_outer_curvature_derivatives {
        None
    } else {
        exact_newton_d2h_many_closure(rho_curvature_scale, hessian_workspace.clone())
    };
    let owned_compute_d2h_many = if use_outer_curvature_derivatives {
        None
    } else {
        exact_newton_d2h_many_closure_owned(rho_curvature_scale, hessian_workspace.clone())
    };

    let efs_eval = joint_outer_evaluate_efs(
        &inner,
        specs,
        &per_block,
        rho_current,
        &beta_flat,
        h_joint_unpen,
        &ranges,
        total,
        ridge,
        moderidge,
        extra_logdet_ridge,
        rho_curvature_scale,
        hessian_logdet_correction,
        include_logdet_h,
        include_logdet_s,
        strict_spd,
        // ψ-bearing EFS path: projected #752 generalized determinant for value
        // and gradient (matched in this single _efs call). Same root-cause fix as
        // the VGH ψ path (gam#808/#787); no batched override here.
        family.use_projected_penalty_logdet(),
        options,
        gam_problem::RhoPrior::Flat,
        family.pseudo_logdet_mode(),
        &compute_dh,
        compute_dh_many.as_deref(),
        &compute_d2h,
        compute_d2h_many.as_deref(),
        Some(owned_compute_dh),
        owned_compute_dh_many,
        Some(owned_compute_d2h),
        owned_compute_d2h_many,
        Some(ext_bundle),
    )
    .map_err(CustomFamilyError::from)?;

    let warm = ConstrainedWarmStart {
        rho: rho_current.clone(),
        block_beta: inner
            .block_states
            .iter()
            .map(|state| state.beta.clone())
            .collect(),
        active_sets: inner.active_sets.clone(),
        cached_inner: Some(cached_inner_mode_from_result(&inner)),
    };

    Ok((efs_eval, warm, inner.converged, inner))
}

/// Evaluate the joint custom-family hyper-surface in fixed-point form for the
/// outer EFS / hybrid-EFS planners.
pub fn evaluate_custom_family_joint_hyper_efs<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
    rho_current: &Array1<f64>,
    hyper_layout: &CustomFamilyHyperLayout,
    warm_start: Option<&CustomFamilyWarmStart>,
) -> Result<CustomFamilyJointHyperEfsResult, CustomFamilyError> {
    Ok(evaluate_custom_family_joint_hyper_efs_owned(
        family,
        specs,
        options,
        rho_current,
        hyper_layout,
        warm_start,
    )?
    .result)
}

/// Evaluate the EFS joint hyperparameter map and retain the exact coefficient
/// mode that produced it.
pub fn evaluate_custom_family_joint_hyper_efs_owned<
    F: CustomFamily + Clone + Send + Sync + 'static,
>(
    family: &F,
    specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
    rho_current: &Array1<f64>,
    hyper_layout: &CustomFamilyHyperLayout,
    warm_start: Option<&CustomFamilyWarmStart>,
) -> Result<CustomFamilyJointHyperEfsOwnedResult, CustomFamilyError> {
    evaluate_custom_family_joint_hyper_efs_owned_shared(
        family,
        specs,
        options,
        rho_current,
        Arc::new(hyper_layout.clone()),
        warm_start,
    )
}

/// Shared-layout variant of
/// [`evaluate_custom_family_joint_hyper_efs_owned`].
pub fn evaluate_custom_family_joint_hyper_efs_owned_shared<
    F: CustomFamily + Clone + Send + Sync + 'static,
>(
    family: &F,
    specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
    rho_current: &Array1<f64>,
    hyper_layout: SharedCustomFamilyHyperLayout,
    warm_start: Option<&CustomFamilyWarmStart>,
) -> Result<CustomFamilyJointHyperEfsOwnedResult, CustomFamilyError> {
    let penalty_counts = validate_blockspecs(specs)?;
    if hyper_layout.block_count() != specs.len() {
        crate::bail_dim_custom!(
            "joint hyper layout block count mismatch: got {}, expected {}",
            hyper_layout.block_count(),
            specs.len()
        );
    }
    let hyper_values = hyper_layout.values().clone();
    let (efs_eval, warm_start, inner_converged, inner) = if hyper_layout.is_empty() {
        outerobjectiveefs(
            family,
            specs,
            options,
            &penalty_counts,
            rho_current,
            warm_start.map(|w| &w.inner),
            gam_problem::RhoPrior::Flat,
        )
        .map_err(CustomFamilyError::from)?
    } else {
        evaluate_custom_family_joint_hyper_efs_internal_shared(
            family,
            specs,
            options,
            &penalty_counts,
            rho_current,
            Arc::clone(&hyper_layout),
            warm_start.map(|w| &w.inner),
        )?
    };
    let mode = CustomFamilyOwnedMode {
        objective: efs_eval.cost,
        rho: warm_start.rho.clone(),
        hyper_values: hyper_values.clone(),
        inner,
    };
    Ok(CustomFamilyJointHyperEfsOwnedResult {
        result: outer_efs_result_to_joint_hyper_efs_result(
            efs_eval,
            warm_start,
            inner_converged,
            hyper_values,
        ),
        mode,
    })
}

/// The derivative-quality floor's degradation rule, pinned on the measured
/// states it was written against (gam#2695, gam#2714, gam#2765).
#[cfg(test)]
mod caller_contract_2695_tests {
    use super::*;
    use gam_problem::{InnerConvergenceTerminalState, JointNewtonTerminalReason};

    fn joint_newton(residual: f64, scale: f64) -> InnerConvergenceTerminalState {
        InnerConvergenceTerminalState::JointNewton {
            cycle: 57,
            stationarity_residual: residual,
            residual_tol: 1.0e-11 * (1.0 + scale),
            stationarity_scale: scale,
            step_inf: 5.354777e-3,
            step_tol: 3.235091e-11,
            resolvable_negative_curvature: true,
            best_stationarity_residual: residual,
            cycles_since_best_residual: 29,
            termination_reason: JointNewtonTerminalReason::CycleBudget,
        }
    }

    /// The survival location-scale seed that refused every fit: it misses the
    /// `1e-11` floor by three orders and is three orders INSIDE the caller's
    /// `1e-6`, so the caller's contract is met and the seed is admissible.
    #[test]
    fn a_seed_inside_the_callers_tolerance_is_admissible_2695() {
        let terminal = joint_newton(6.952333e0, 3.353363e9);
        assert!(joint_newton_meets_caller_inner_contract(
            Some(&terminal),
            1.0e-6
        ));
        assert!(
            !joint_newton_meets_caller_inner_contract(Some(&terminal), 1.0e-11),
            "the floor itself is missed — that is what makes this a degradation and not a no-op"
        );
    }

    /// The #2765 replay's diverged probe: `scale = 2.5e16` with a residual of
    /// `2.7e17` is seven orders OUTSIDE the caller's contract too, so the
    /// degradation must not rescue it.
    #[test]
    fn a_diverged_solve_is_still_refused_at_the_callers_tolerance_2765() {
        let terminal = joint_newton(2.729331e17, 2.481211e16);
        assert!(!joint_newton_meets_caller_inner_contract(
            Some(&terminal),
            1.0e-6
        ));
    }

    /// Every state that carries no joint-Newton residual to judge keeps the
    /// refusal: absence of a measurement is not a passing measurement.
    #[test]
    fn a_state_without_a_joint_newton_residual_keeps_the_refusal_2695() {
        assert!(!joint_newton_meets_caller_inner_contract(None, 1.0e-6));
        let blockwise = InnerConvergenceTerminalState::Blockwise {
            cycle: 3,
            max_accepted_step: 1.0e-9,
            max_proposed_step: 1.0e-3,
            step_tol: 1.0e-10,
            objective_change: 1.0e-12,
            objective_tol: 1.0e-9,
            joint_stationarity_ok: false,
        };
        assert!(!joint_newton_meets_caller_inner_contract(
            Some(&blockwise),
            1.0e-6
        ));
        let non_finite = joint_newton(f64::NAN, 3.353363e9);
        assert!(!joint_newton_meets_caller_inner_contract(
            Some(&non_finite),
            1.0e-6
        ));
        let unbounded_scale = joint_newton(6.952333e0, f64::INFINITY);
        assert!(!joint_newton_meets_caller_inner_contract(
            Some(&unbounded_scale),
            1.0e-6
        ));
        let terminal = joint_newton(6.952333e0, 3.353363e9);
        assert!(!joint_newton_meets_caller_inner_contract(
            Some(&terminal),
            f64::NAN
        ));
    }
}
