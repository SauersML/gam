//! Builds the unified ψ `HyperCoord` objects + pair/drift callbacks from the
//! family-provided penalty derivatives, and evaluates the custom-family
//! hyper-objective (joint and EFS variants, shared + public entry points).

use super::*;

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
) -> Result<Vec<HyperCoord>, String> {
    let ranges = block_param_ranges(specs);
    let total = beta_flat.len();
    let per_block = split_log_lambdas(&Array1::from_vec(rho.to_vec()), penalty_counts)?;
    let per_block_lambdas =
        exact_lambdas_by_block(&per_block, "psi hyper log strength").map_err(String::from)?;

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
        return Err(format!(
            "custom-family hyper workspace returned {} first-order axes for layout length {total_axes}",
            terms.len()
        ));
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
    let jeffreys_hphi_ctx: Option<(Array2<f64>, Array2<f64>)> = if family
        .joint_jeffreys_term_required()
        && !hyper_layout.is_empty()
    {
        match (
            build_joint_jeffreys_subspace(specs, &ranges)?,
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

    for psi_global in 0..total_axes {
        let axis = hyper_layout
            .axis(psi_global)
            .ok_or_else(|| format!("missing typed hyper axis {psi_global}"))?;
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
            family.exact_newton_joint_psi_terms(
                synced_states,
                specs,
                hyper_layout,
                psi_global,
            )?
        };
        let psi_terms = match (axis, psi_terms) {
            (_, Some(terms)) => terms,
            (CustomFamilyHyperAxis::DesignPenalty { .. }, None) => {
                ExactNewtonJointPsiTerms::zeros(total)
            }
            (CustomFamilyHyperAxis::Family { family_axis }, None) => {
                return Err(format!(
                    "family-owned hyper axis {family_axis} has no exact first-order V_i/g_i/H_i terms"
                ));
            }
        };

        // 2. Assemble generic penalty motion only for a typed design/penalty
        // axis. Family axes have no fabricated block owner and therefore carry
        // exactly zero S_i.
        let penalty_motion = hyper_layout.design_derivative(psi_global).map(
            |(block_idx, _, deriv)| {
                let (start, end) = ranges[block_idx];
                let p_block = end - start;
                let s_psi_local = assemble_block_local_s_psi(
                    deriv,
                    &per_block_lambdas[block_idx],
                    p_block,
                );
                (block_idx, start, end, s_psi_local)
            },
        );

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
            if let (Some((z_j, h_joint)), Some(pert_info)) =
                (jeffreys_hphi_ctx.as_ref(), firth_pert_info.as_ref())
            {
                let phi_psi =
                    gam_solve::estimate::reml::jeffreys_subspace::joint_jeffreys_phi_explicit_param_derivative(
                        h_joint.view(),
                        z_j.view(),
                        pert_info,
                    )?;
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
            if let (Some((z_j, h_joint)), Some(pert_info)) =
                (jeffreys_hphi_ctx.as_ref(), firth_pert_info.as_ref())
            {
                for a_idx in 0..total {
                    let mut e_a = Array1::<f64>::zeros(total);
                    e_a[a_idx] = 1.0;
                    let hdot_a = family
                        .joint_jeffreys_information_directional_derivative_with_specs(
                            synced_states,
                            specs,
                            &e_a,
                        )?;
                    let psi_hdot_a = family.exact_newton_joint_psihessian_directional_derivative(
                        synced_states,
                        specs,
                        hyper_layout,
                        psi_global,
                        &e_a,
                    )?;
                    if let (Some(hdot_a), Some(psi_hdot_a)) = (hdot_a, psi_hdot_a) {
                        let phi_psi_beta_a =
                            gam_solve::estimate::reml::jeffreys_subspace::joint_jeffreys_phi_explicit_param_second_derivative(
                                h_joint.view(),
                                z_j.view(),
                                pert_info,
                                &hdot_a,
                                &psi_hdot_a,
                            )?;
                        g[a_idx] -= phi_psi_beta_a;
                    }
                }
            }
        let ld_s = match (s_logdet_blocks, penalty_motion.as_ref()) {
            (Some(blocks), Some((block_idx, _, _, s_psi_local))) => {
                blocks[*block_idx].tau_gradient_component(s_psi_local)
            }
            _ => 0.0,
        };

        // Build drift: use block-local representation when possible to avoid
        // materializing full p×p dense matrices.
        let drift = if let Some(operator) = psi_terms.hessian_psi_operator {
            if let Some((_, start, end, s_psi_local)) = penalty_motion {
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
            }
        } else {
            // Dense Hessian term exists (e.g., from non-implicit family).
            // Add block-local penalty motion only for DesignPenalty axes.
            let mut dense_b = psi_terms.hessian_psi;
            if let Some((_, start, end, s_psi_local)) = penalty_motion {
                dense_b
                    .slice_mut(ndarray::s![start..end, start..end])
                    .scaled_add(1.0, &s_psi_local);
            }
                // `dense_b` is now `∂_ρ_i H_joint|_β`. Add the explicit Jeffreys term
                // `∂_ρ_i H_Φ|_β` (gam#854) using it as the H_joint perturbation, the
                // family's base directional Hessian derivative `Hdot[e_a]`, and the
                // ψ-Hessian directional derivative `∂_ρ_i Hdot[e_a]|_β`. The helper
                // returns zeros when the conditioning gate skips the term or the
                // family lacks the exact directional derivatives, so a clean /
                // well-conditioned fit is byte-unchanged.
                if let Some((z_j, h_joint)) = jeffreys_hphi_ctx
                    .as_ref()
                    .filter(|_| jeffreys_info_depends_on_psi)
                {
                    let explicit_hphi =
                        gam_solve::estimate::reml::jeffreys_subspace::joint_jeffreys_hphi_explicit_param_derivative(
                            h_joint.view(),
                            z_j.view(),
                            &dense_b,
                            |dir: &Array1<f64>| {
                                family.joint_jeffreys_information_directional_derivative_with_specs(
                                    synced_states,
                                    specs,
                                    dir,
                                )
                            },
                            |dir: &Array1<f64>| {
                                family.exact_newton_joint_psihessian_directional_derivative(
                                    synced_states,
                                    specs,
                                    hyper_layout,
                                    psi_global,
                                    dir,
                                )
                            },
                        )?;
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
pub fn build_jeffreys_hphi_ctx<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    synced_states: &[ParameterBlockState],
    specs: &[ParameterBlockSpec],
    hyper_layout: &CustomFamilyHyperLayout,
    total: usize,
) -> Result<Option<(Array2<f64>, Array2<f64>)>, String> {
    if family.joint_jeffreys_term_required() && !hyper_layout.is_empty() {
        let ranges = block_param_ranges(specs);
        Ok(
            match (
                build_joint_jeffreys_subspace(specs, &ranges)?,
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
) -> Result<Option<ContractedPsiSecondOrderFn>, String> {
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
        exact_lambdas_by_block(&per_block, "contracted psi log strength").map_err(String::from)?,
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
            return Err(format!(
                "contracted ψψ hook basis probe shape mismatch at axis {axis_idx}: \
                 objective={}, score={}x{}, hessian={}, psi_dim={psi_dim}, beta_dim={total}",
                terms.objective.len(),
                terms.score.nrows(),
                terms.score.ncols(),
                terms.hessian.len(),
            ));
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
    let firth_ctx: Option<(Arc<Array2<f64>>, Arc<Array2<f64>>, Arc<Vec<Array2<f64>>>)> =
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
                            Some((Arc::new(z_j), Arc::new(h_joint), Arc::new(pert_first)))
                        } else {
                            None
                        }
                    }
                    None => None,
                }
            }
            _ => None,
        };

    let hook = move |alpha_psi: &[f64]| -> Result<Option<ContractedPsiSecondOrder>, String> {
        if alpha_psi.len() != psi_dim {
            return Err(format!(
                "contracted ψψ hook: alpha_psi length {} != psi_dim {psi_dim}",
                alpha_psi.len()
            ));
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
            return Err(format!(
                "contracted ψψ hook: family kernel shape mismatch (objective={}, score={}x{}, hessian={}, psi_dim={psi_dim}, beta_dim={total})",
                objective.len(),
                score.nrows(),
                score.ncols(),
                hessian.len(),
            ));
        }

        for i in 0..psi_dim {
            // EXPLICIT Firth/Jeffreys ψψ VALUE second derivative (gam#1607):
            //   objective[i] -= ∂_{ψ_i}∂_{ψ(α)}Φ.
            // This applies to both design/penalty and family-owned axes.
            if let Some((z_j, h_joint, pert_first)) = firth_ctx.as_ref() {
                let pert_i_alpha = match &hessian[i] {
                    DriftDerivResult::Dense(m) => m.clone(),
                    DriftDerivResult::Operator(op) => op.mul_mat(&Array2::<f64>::eye(total)),
                };
                let mut pert_alpha = Array2::<f64>::zeros((total, total));
                for (j, &aj) in alpha_psi.iter().enumerate() {
                    if aj != 0.0 {
                        pert_alpha.scaled_add(aj, &pert_first[j]);
                    }
                }
                let phi_psi_psi =
                    gam_solve::estimate::reml::jeffreys_subspace::joint_jeffreys_phi_explicit_param_second_derivative(
                        h_joint.view(),
                        z_j.view(),
                        &pert_first[i],
                        &pert_alpha,
                        &pert_i_alpha,
                    )?;
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
                let deriv_i =
                    &hyper_layout.design_derivative_blocks()[axis_i.block][axis_i.local];
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

    Ok(Some(Arc::new(hook) as ContractedPsiSecondOrderFn))
}

/// Build pair callbacks for ψ-ψ and ρ-ψ Hessian entries.
///
/// Returns two closures:
///
/// 1. **ext-ext** `(psi_i, psi_j) -> Result<HyperCoordPair, String>`: second-order
///    fixed-β objects for a pair of ψ coordinates.
///
/// 2. **rho-ext** `(rho_k, psi_j) -> Result<HyperCoordPair, String>`: mixed second-order
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
        Box<dyn Fn(usize, usize) -> Result<HyperCoordPair, String> + Send + Sync>,
        Box<dyn Fn(usize, usize) -> Result<HyperCoordPair, String> + Send + Sync>,
    ),
    String,
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
            .map_err(String::from)?,
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
    let firth_pair_ctx: Option<(Arc<Array2<f64>>, Arc<Array2<f64>>, Arc<Vec<Array2<f64>>>)> =
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
                    return Err(format!(
                        "custom-family hyper workspace returned {} first-order axes for layout length {psi_dim}",
                        all.len()
                    ));
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
                    Some((Arc::new(z_j), Arc::new(h_joint), Arc::new(pert_first)))
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
        .map_err(|err| format!("rho-penalty dense cache refused by memory governor: {err}"))?;
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
        vec![vec![None::<ExactNewtonJointPsiSecondOrderTerms>; hyper_layout.len()]; hyper_layout.len()];
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
                format!(
                    "typed family hyper pair ({i}, {j}) has no exact V_ij/g_ij/H_ij terms"
                )
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
            move |psi_i: usize, psi_j: usize| -> Result<HyperCoordPair, String> {
            if psi_i >= hyper_layout.len() || psi_j >= hyper_layout.len() {
                return Err(format!(
                    "typed hyper pair index out of bounds: ({psi_i}, {psi_j}) for {} axes",
                    hyper_layout.len()
                ));
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
                return Err(format!(
                    "typed hyper pair ({psi_i}, {psi_j}) returned score length {}, expected {total}",
                    g.len()
                ));
            }
            if b_mat.dim() != (0, 0) && b_mat.dim() != (total, total) {
                return Err(format!(
                    "typed hyper pair ({psi_i}, {psi_j}) returned dense Hessian shape {:?}, expected (0, 0) or ({total}, {total})",
                    b_mat.dim()
                ));
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
            if let Some((z_j, h_joint, pert_first)) = firth_pair_ctx.as_ref()
                && psi_i < pert_first.len()
                && psi_j < pert_first.len()
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
                    let phi_psi_psi =
                        gam_solve::estimate::reml::jeffreys_subspace::joint_jeffreys_phi_explicit_param_second_derivative(
                            h_joint.view(),
                            z_j.view(),
                            &pert_first[psi_i],
                            &pert_first[psi_j],
                            &pert_ij,
                        )
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
                    let mut b_local =
                        b_mat.slice_mut(s![cache_i.start..cache_i.end, cache_i.start..cache_i.end]);
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
                        }) as Arc<dyn HyperOperator>,
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
        }) as Box<dyn Fn(usize, usize) -> Result<HyperCoordPair, String> + Send + Sync>
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
            move |rho_k: usize, psi_j: usize| -> Result<HyperCoordPair, String> {
            if rho_k >= rho_penalty_cache.len() || psi_j >= hyper_layout.len() {
                return Err(format!(
                    "rho×typed-hyper pair index out of bounds: ({rho_k}, {psi_j}) for {} rho and {} non-rho axes",
                    rho_penalty_cache.len(),
                    hyper_layout.len()
                ));
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
                    let ds_k_dpsi = if lambda_k.abs() > 1e-300 {
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
        }) as Box<dyn Fn(usize, usize) -> Result<HyperCoordPair, String> + Send + Sync>
    };

    Ok((ext_ext, rho_ext))
}

/// Build the M_i[u] = D_β B_i[u] callback for ψ coordinates.
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
) -> Result<Option<FixedDriftDerivFn>, String> {
    if hessian_beta_independent {
        // Likelihood Hessian is β-independent; M_i ≡ 0.
        return Ok(None);
    }

    if hyper_layout.family_axis_count() != 0 && psi_workspace.is_none() {
        return Err(
            "family-owned hyper axes require one owned exact-psi workspace for directional Hessian drift"
                .to_string(),
        );
    }

    let synced_arc = Arc::new(synced_states.to_vec());
    let specs_arc = Arc::new(specs.to_vec());
    let family_arc = Arc::new(family.clone());
    let psi_workspace = psi_workspace;

    Ok(Some(Box::new(
        move |ext_idx: usize,
              direction: &Array1<f64>|
              -> Result<Option<DriftDerivResult>, String> {
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
                None if hyper_layout.family_axis(ext_idx).is_some() => Err(format!(
                    "family-owned hyper axis {ext_idx} has no exact D_beta H_i[u] term for the requested direction"
                )),
                None => Ok(None),
            }
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
    // evaluations already have their inner tolerance managed deliberately by
    // `derivative_quality_options_and_warm_start` (which intentionally LOOSENS
    // it for large-scale ψ fits), and must not be re-tightened here.
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
    if !inner.converged {
        let theta_dim = rho_dim + psi_dim;
        return Err(CustomFamilyError::UnsupportedConfiguration {
            reason: format!(
                "custom-family inner solve did not converge after {} cycle(s); \
             refusing to expose profile objective derivatives for theta_dim={} \
             (rho_dim={}, psi_dim={}). The analytic outer gradient/Hessian \
             require the inner KKT equation F_beta(beta, theta)=0; returning \
             a value with zero or shape-only derivatives is mathematically \
             inconsistent.",
                inner.cycles, theta_dim, rho_dim, psi_dim
            ),
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
            let block_results: Vec<Result<PenaltyPseudologdet, String>> = (0..specs.len())
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
                })
                .collect();
            let blocks: Result<Vec<_>, _> = block_results.into_iter().collect();
            Some(blocks?)
        } else {
            None
        };

        let robust_jeffreys_hphi =
            custom_family_outer_jeffreys_hphi(
                family,
                &inner.block_states,
                specs,
                &ranges,
                eval_mode,
            )?;
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
                        |_direction: &Array1<f64>| -> Result<Option<DriftDerivResult>, String> {
                            Ok(None)
                        };
                    let no_d2h = |_u: &Array1<f64>,
                                  _v: &Array1<f64>|
                     -> Result<Option<DriftDerivResult>, String> {
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
                        gradient,
                        outer_hessian: gam_problem::HessianValue::Unavailable,
                        warm_start: value_only.warm_start,
                        inner_converged: inner.converged,
                        hyper_values: hyper_layout.values().clone(),
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
                eval_mode,
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
        custom_family_outer_jeffreys_hphi(
            family,
            &inner.block_states,
            specs,
            &ranges,
            eval_mode,
        )?;
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
                        gradient,
                        outer_hessian: gam_problem::HessianValue::Unavailable,
                        warm_start: value_only.warm_start,
                        inner_converged: inner.converged,
                        hyper_values: hyper_layout.values().clone(),
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
                eval_mode,
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
    let compute_dh = |direction: &Array1<f64>| -> Result<Option<DriftDerivResult>, String> {
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
                    ) }.into()),
                }
            }
            BlockWorkingSet::Diagonal {
                working_response: _,
                working_weights,
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
                    }
                    .into());
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
     -> Result<Option<DriftDerivResult>, String> {
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
                    ) }.into()),
                }
            }
            BlockWorkingSet::Diagonal {
                working_response: _,
                working_weights: _,
            } => {
                let x_dyn = diagonal_design.as_ref().ok_or_else(|| {
                    format!("missing dynamic design for block {b} diagonal second correction")
                })?;
                let x_dense = x_dyn.to_dense();
                let n = x_dense.nrows();

                let reject_second_order_geometry = |label: &str,
                                                    geom: Option<
                    BlockGeometryDirectionalDerivative,
                >|
                 -> Result<(), String> {
                    if let Some(geom_dir) = geom {
                        let has_offset = geom_dir.d_offset.iter().any(|value| *value != 0.0);
                        if geom_dir.d_design.is_some() || has_offset {
                            return Err(CustomFamilyError::UnsupportedConfiguration { reason: format!(
                                "block {b} diagonal d2H requires second-order block-geometry derivatives for {label}; use an exact-newton or joint outer path"
                            ) }.into());
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
                    }
                    .into());
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
            eval_mode,
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
pub fn evaluate_custom_family_joint_hyper_owned<
    F: CustomFamily + Clone + Send + Sync + 'static,
>(
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
    Ok(outer_eval_result_into_joint_hyper_owned_result(
        eval_result,
    ))
}

pub fn evaluate_custom_family_joint_hyper_shared<
    F: CustomFamily + Clone + Send + Sync + 'static,
>(
    family: &F,
    specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
    rho_current: &Array1<f64>,
    hyper_layout: SharedCustomFamilyHyperLayout,
    warm_start: Option<&CustomFamilyWarmStart>,
    eval_mode: EvalMode,
) -> Result<CustomFamilyJointHyperResult, CustomFamilyError> {
    Ok(evaluate_custom_family_joint_hyper_owned_shared(
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

/// Shared-layout variant of
/// [`evaluate_custom_family_joint_hyper_owned`].
pub fn evaluate_custom_family_joint_hyper_owned_shared<
    F: CustomFamily + Clone + Send + Sync + 'static,
>(
    family: &F,
    specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
    rho_current: &Array1<f64>,
    hyper_layout: SharedCustomFamilyHyperLayout,
    warm_start: Option<&CustomFamilyWarmStart>,
    eval_mode: EvalMode,
) -> Result<CustomFamilyJointHyperOwnedResult, CustomFamilyError> {
    let penalty_counts = validate_blockspecs(specs)?;
    let has_psi_derivatives = !hyper_layout.is_empty();
    let (eval_options, strict_warm_start) =
        derivative_quality_options_and_warm_start(options, warm_start, has_psi_derivatives);
    let eval_result = evaluate_custom_family_hyper_internal_shared(
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
        eval_mode,
        None,
    )?;
    Ok(outer_eval_result_into_joint_hyper_owned_result(
        eval_result,
    ))
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
    let screened_objective_bits = screened_winner.objective.to_bits();
    let selected_inner = screened_winner.inner;
    let (eval_options, _) =
        derivative_quality_options_and_warm_start(options, None, has_psi_derivatives);
    let derivative_eval = evaluate_custom_family_hyper_internal_shared(
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
    if derivative_eval.objective.to_bits() != screened_objective_bits {
        return Err(CustomFamilyError::UnsupportedConfiguration {
            reason: format!(
                "best coefficient-mode candidate {selected_candidate} changed profile objective between value screening and derivative assembly"
            ),
        });
    }
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
    const DIRECT_JOINT_HYPER_INNER_TOL_FLOOR: f64 = 1e-10;
    const DIRECT_JOINT_HYPER_MIN_CYCLES: usize = 200;

    let mut eval_options = options.clone();
    // The alignment exists so exact joint-hyper evaluations with real ψ
    // coordinates resolve the inner solve at the outer optimizer's requested
    // derivative scale. With zero ψ-derivative blocks this API is just the
    // rho-only outer surface; mutating its inner tolerance makes the direct
    // joint-hyper path evaluate a different function than the rho-only path.
    if !has_psi_derivatives {
        return (eval_options, None);
    }
    //
    // Do not hard-force f64-precision KKT solves for every ψ-bearing model:
    // large-scale survival marginal-slope fits have row-summed objectives
    // around 1e5-1e6, so `1e-10 * objective` asks the inner loop to resolve
    // gradient components far below the outer optimizer's own `outer_tol`.
    // Matching the inner target to the outer target keeps the IFT gradient
    // noise below the requested optimization accuracy without rejecting all
    // startup seeds after hundreds of accepted but numerically flat Newton
    // steps.
    let default_inner_tol = BlockwiseFitOptions::default().inner_tol;
    let requested_tighter_than_default = eval_options.inner_tol < default_inner_tol;
    let direct_joint_hyper_inner_tol = if requested_tighter_than_default {
        eval_options.inner_tol.max(1.0e-12)
    } else {
        eval_options
            .outer_tol
            .max(DIRECT_JOINT_HYPER_INNER_TOL_FLOOR)
    };
    let tolerance_differs = eval_options.inner_tol != direct_joint_hyper_inner_tol;
    let tightening = eval_options.inner_tol > direct_joint_hyper_inner_tol;
    let align = eval_options.inner_max_cycles > 1 && tolerance_differs;
    let psi_safe_warm_start = warm_start_without_cached_inner_for_psi_derivatives(
        warm_start.map(|warm| &warm.inner),
        true,
    )
    .map(|inner| CustomFamilyWarmStart { inner });
    if !align {
        return (eval_options, psi_safe_warm_start);
    }
    eval_options.inner_tol = direct_joint_hyper_inner_tol;
    if tightening {
        eval_options.inner_max_cycles = eval_options
            .inner_max_cycles
            .max(DIRECT_JOINT_HYPER_MIN_CYCLES);
    }
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
            include_logdet_h,
            include_logdet_s,
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
        let block_results: Vec<Result<PenaltyPseudologdet, String>> = (0..specs.len())
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

pub fn evaluate_custom_family_joint_hyper_efs_shared<
    F: CustomFamily + Clone + Send + Sync + 'static,
>(
    family: &F,
    specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
    rho_current: &Array1<f64>,
    hyper_layout: SharedCustomFamilyHyperLayout,
    warm_start: Option<&CustomFamilyWarmStart>,
) -> Result<CustomFamilyJointHyperEfsResult, CustomFamilyError> {
    Ok(evaluate_custom_family_joint_hyper_efs_owned_shared(
        family,
        specs,
        options,
        rho_current,
        hyper_layout,
        warm_start,
    )?
    .result)
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
    let (efs_eval, warm_start, inner_converged, inner) =
        if hyper_layout.is_empty() {
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
