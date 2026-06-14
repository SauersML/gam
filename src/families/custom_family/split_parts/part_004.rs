
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
    derivative_blocks: &[Vec<CustomFamilyBlockPsiDerivative>],
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

    let mut coords = Vec::new();
    let mut psi_global = 0usize;

    let build_psi_hyper_coords_start = std::time::Instant::now();
    let total_axes: usize = derivative_blocks.iter().map(|b| b.len()).sum();

    let batched_terms: Option<Vec<ExactNewtonJointPsiTerms>> = match psi_workspace.as_ref() {
        Some(workspace) => workspace.first_order_terms_all()?,
        None => None,
    };

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
        && derivative_blocks.iter().any(|block| !block.is_empty())
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

    for (block_idx, block_derivs) in derivative_blocks.iter().enumerate() {
        let (start, end) = ranges[block_idx];
        let p_block = end - start;

        for deriv in block_derivs.iter() {
            // 1. Get family-provided likelihood objects (joint flattened space).
            let psi_terms = if let Some(batched) = batched_terms.as_ref() {
                batched[psi_global].clone()
            } else if let Some(workspace) = psi_workspace.as_ref() {
                if let Some(terms) = workspace.first_order_terms(psi_global)? {
                    terms
                } else {
                    family
                        .exact_newton_joint_psi_terms(
                            synced_states,
                            specs,
                            derivative_blocks,
                            psi_global,
                        )?
                        .unwrap_or_else(|| ExactNewtonJointPsiTerms::zeros(total))
                }
            } else {
                family
                    .exact_newton_joint_psi_terms(
                        synced_states,
                        specs,
                        derivative_blocks,
                        psi_global,
                    )?
                    .unwrap_or_else(|| ExactNewtonJointPsiTerms::zeros(total))
            };

            // 2. Assemble S_ψ from penalty derivatives (block-local, not embedded).
            let s_psi_local = assemble_block_local_s_psi(deriv, &per_block[block_idx], p_block);

            // 3. Build HyperCoord using block-local S_ψ (avoids full p×p materialization).
            let beta_block = beta_flat.slice(ndarray::s![start..end]);
            let s_psi_beta_local = s_psi_local.dot(&beta_block);
            let a = psi_terms.objective_psi + 0.5 * beta_block.dot(&s_psi_beta_local);
            // Embed s_psi_beta into full p-vector for the score.
            let mut s_psi_beta = Array1::zeros(total);
            s_psi_beta
                .slice_mut(ndarray::s![start..end])
                .assign(&s_psi_beta_local);
            let g = &psi_terms.score_psi + &s_psi_beta;
            let ld_s = if let Some(blocks) = s_logdet_blocks {
                blocks[block_idx].tau_gradient_component(&s_psi_local)
            } else {
                0.0
            };

            // Build drift: use block-local representation when possible to avoid
            // materializing full p×p dense matrices.
            let drift = if psi_terms.hessian_psi_operator.is_some() {
                // No dense Hessian contribution — penalty is block-local, operator
                // (if present) handles the likelihood part. O(p_block²) fast path.
                HyperCoordDrift::from_block_local_and_operator(
                    s_psi_local,
                    start,
                    end,
                    total,
                    psi_terms.hessian_psi_operator,
                )
            } else {
                // Dense Hessian term exists (e.g., from non-implicit family).
                // Must add block-local penalty into the dense matrix.
                let mut dense_b = psi_terms.hessian_psi;
                dense_b
                    .slice_mut(ndarray::s![start..end, start..end])
                    .scaled_add(1.0, &s_psi_local);
                // `dense_b` is now `∂_ρ_i H_joint|_β`. Add the explicit Jeffreys term
                // `∂_ρ_i H_Φ|_β` (gam#854) using it as the H_joint perturbation, the
                // family's base directional Hessian derivative `Hdot[e_a]`, and the
                // ψ-Hessian directional derivative `∂_ρ_i Hdot[e_a]|_β`. The helper
                // returns zeros when the conditioning gate skips the term or the
                // family lacks the exact directional derivatives, so a clean /
                // well-conditioned fit is byte-unchanged.
                if let Some((z_j, h_joint)) = jeffreys_hphi_ctx.as_ref() {
                    let explicit_hphi =
                        crate::estimate::reml::jeffreys_subspace::joint_jeffreys_hphi_explicit_param_derivative(
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
                                    derivative_blocks,
                                    psi_global,
                                    dir,
                                )
                            },
                        )?;
                    dense_b += &explicit_hphi;
                }
                HyperCoordDrift::from_parts(Some(dense_b), psi_terms.hessian_psi_operator)
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

            psi_global += 1;
        }
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
/// combined-direction likelihood kernel (`second_order_terms_contracted`);
/// otherwise `None`, which keeps the outer-Hessian operator on the exact
/// per-pair `ext_ext_fn` assembly.
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
pub(crate) fn build_contracted_psi_hook(
    specs: &[ParameterBlockSpec],
    derivative_blocks: SharedDerivativeBlocks,
    beta_flat: &Array1<f64>,
    rho: &[f64],
    penalty_counts: &[usize],
    s_logdet_blocks: Option<&[PenaltyPseudologdet]>,
    psi_workspace: Option<Arc<dyn ExactNewtonJointPsiWorkspace>>,
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
    let beta_arc = Arc::new(beta_flat.clone());
    let ranges_arc = Arc::new(ranges);
    let s_logdet_block_cache = Arc::new(s_logdet_blocks.map(|blocks| blocks.to_vec()));

    // ψ → (block, local) location and block-local S_ψ for every ψ axis, built
    // once. `s_local` (block-local S_ψ) is reused for the τ-Hessian and as the
    // first leg of the bilinear `tr(S⁺ S_ψi S⁺ S_ψj)` penalty-logdet term.
    struct PsiAxis {
        block: usize,
        local: usize,
        start: usize,
        end: usize,
        s_psi_local: Array2<f64>,
    }
    let mut axes: Vec<PsiAxis> = Vec::new();
    for (block_idx, block_derivs) in derivative_blocks.iter().enumerate() {
        let (start, end) = ranges_arc[block_idx];
        let p_block = end - start;
        for (local_idx, deriv) in block_derivs.iter().enumerate() {
            let s_psi_local = assemble_block_local_s_psi(deriv, &per_block[block_idx], p_block);
            axes.push(PsiAxis {
                block: block_idx,
                local: local_idx,
                start,
                end,
                s_psi_local,
            });
        }
    }
    let axes = Arc::new(axes);
    let psi_dim = axes.len();
    if psi_dim == 0 {
        return Ok(None);
    }

    let derivative_blocks = Arc::clone(&derivative_blocks);

    let hook = move |alpha_psi: &[f64]| -> Result<Option<ContractedPsiSecondOrder>, String> {
        if alpha_psi.len() != psi_dim {
            return Err(format!(
                "contracted ψψ hook: alpha_psi length {} != psi_dim {psi_dim}",
                alpha_psi.len()
            ));
        }
        // Family likelihood ψψ contraction (one combined-direction row pass).
        // Declining here (e.g. a σ-aux axis carried weight) declines the whole
        // hook so the operator builder keeps the per-pair assembly.
        let Some(likelihood) = workspace.second_order_terms_contracted(alpha_psi)? else {
            return Ok(None);
        };
        let mut objective = likelihood.objective;
        let mut score = likelihood.score;
        let mut ld_s = Array1::<f64>::zeros(psi_dim);
        // Per-output-row penalty drift `Σ_j α_j S_{ψi ψj}` (block-local),
        // composed onto the likelihood `hessian[i]` operator below.
        let mut hessian: Vec<DriftDerivResult> = likelihood.hessian;
        if objective.len() != psi_dim || score.nrows() != psi_dim || hessian.len() != psi_dim {
            return Err(format!(
                "contracted ψψ hook: family kernel shape mismatch (objective={}, score_rows={}, hessian={}, psi_dim={psi_dim})",
                objective.len(),
                score.nrows(),
                hessian.len(),
            ));
        }

        for (i, axis_i) in axes.iter().enumerate() {
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
                let aj = alpha_psi[j];
                if aj == 0.0 || axis_j.block != axis_i.block {
                    continue;
                }
                let deriv_i = &derivative_blocks[axis_i.block][axis_i.local];
                let s_ij = assemble_block_local_s_psi_psi(
                    deriv_i,
                    axis_j.local,
                    &per_block[axis_i.block],
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
            let block_drift: Arc<dyn HyperOperator> =
                Arc::new(crate::solver::estimate::reml::unified::BlockLocalDrift {
                    local: s_psi_psi_alpha.clone(),
                    start: axis_i.start,
                    end: axis_i.end,
                    total_dim: total,
                });
            let combined = match std::mem::replace(
                &mut hessian[i],
                DriftDerivResult::Operator(Arc::clone(&block_drift)),
            ) {
                DriftDerivResult::Operator(existing) => DriftDerivResult::Operator(Arc::new(
                    crate::solver::estimate::reml::unified::CompositeHyperOperator {
                        dense: None,
                        operators: vec![existing, block_drift],
                        dim_hint: total,
                    },
                )),
                DriftDerivResult::Dense(dense) => DriftDerivResult::Operator(Arc::new(
                    crate::solver::estimate::reml::unified::CompositeHyperOperator {
                        dense: Some(dense),
                        operators: vec![block_drift],
                        dim_hint: total,
                    },
                )),
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
/// 1. **ext-ext** `(psi_i, psi_j) -> HyperCoordPair`: second-order
///    fixed-β objects for a pair of ψ coordinates.
///
/// 2. **rho-ext** `(rho_k, psi_j) -> HyperCoordPair`: mixed second-order
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
/// * `derivative_blocks` - Per-block ψ derivative payloads.
/// * `beta_flat` - Flattened joint coefficient vector at the inner mode.
/// * `rho` - Current log-smoothing parameters (flat).
/// * `penalty_counts` - Number of penalties per block.
/// * `s_logdet_blocks` - Optional exact block-local pseudologdet eigenspaces.
pub fn build_psi_pair_callbacks<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    synced_states: &[ParameterBlockState],
    specs: &[ParameterBlockSpec],
    derivative_blocks: SharedDerivativeBlocks,
    beta_flat: &Array1<f64>,
    rho: &[f64],
    penalty_counts: &[usize],
    s_logdet_blocks: Option<&[PenaltyPseudologdet]>,
    psi_workspace: Option<Arc<dyn ExactNewtonJointPsiWorkspace>>,
) -> Result<
    (
        Box<dyn Fn(usize, usize) -> HyperCoordPair + Send + Sync>,
        Box<dyn Fn(usize, usize) -> HyperCoordPair + Send + Sync>,
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
    let specs_arc = Arc::new(specs.to_vec());
    let beta_arc = Arc::new(beta_flat.clone());
    let synced_arc = Arc::new(synced_states.to_vec());
    let ranges_arc = Arc::new(ranges);
    let family_arc = Arc::new(family.clone());

    let s_logdet_block_cache = Arc::new(s_logdet_blocks.map(|blocks| blocks.to_vec()));

    struct PsiPenaltyCacheEntry {
        block_idx: usize,
        local_idx: usize,
        start: usize,
        end: usize,
        /// Block-local S_ψ matrix, stored for use with `PenaltyPseudologdet` methods.
        s_local: Option<Array2<f64>>,
    }

    struct RhoPenaltyCacheEntry {
        block_idx: usize,
        penalty_idx: usize,
        start: usize,
        end: usize,
        /// Unscaled penalty matrix S_k for use with `PenaltyPseudologdet::rho_tau_hessian_component`.
        s_k_unscaled: Array2<f64>,
    }

    // Build the psi coordinate cache once. These block-local S_psi matrices are
    // reused by ψψ and ρψ callbacks, avoiding repeated assembly inside the
    // O(q²) ext-ext loop.
    let mut psi_penalty_cache: Vec<PsiPenaltyCacheEntry> = Vec::new();
    for (block_idx, block_derivs) in derivative_blocks.iter().enumerate() {
        let (start, end) = ranges_arc[block_idx];
        let p_block = end - start;
        for (local_idx, deriv) in block_derivs.iter().enumerate() {
            let s_local = assemble_block_local_s_psi(deriv, &per_block[block_idx], p_block);
            // Store the block-local S_ψ matrix when penalty logdet is active;
            // PenaltyPseudologdet methods will handle pseudoinverse and leakage internally.
            let s_local_opt = if s_logdet_block_cache.is_some() {
                Some(s_local)
            } else {
                None
            };
            psi_penalty_cache.push(PsiPenaltyCacheEntry {
                block_idx,
                local_idx,
                start,
                end,
                s_local: s_local_opt,
            });
        }
    }
    let psi_penalty_cache = Arc::new(psi_penalty_cache);

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
    let rho_penalty_cache = Arc::new(rho_penalty_cache);

    // ψ-ψ pair callback
    let ext_ext = {
        let per_block = Arc::clone(&per_block);
        let derivative_blocks = Arc::clone(&derivative_blocks);
        let specs_arc = Arc::clone(&specs_arc);
        let beta_arc = Arc::clone(&beta_arc);
        let synced_arc = Arc::clone(&synced_arc);
        let s_logdet_block_cache = Arc::clone(&s_logdet_block_cache);
        let psi_penalty_cache = Arc::clone(&psi_penalty_cache);
        let family_arc = Arc::clone(&family_arc);
        let psi_workspace = psi_workspace.clone();

        Box::new(move |psi_i: usize, psi_j: usize| -> HyperCoordPair {
            // Defensive bounds check: callers in the unified outer solver only ever
            // pass indices in `0..psi_penalty_cache.len()`, but treating an OOB
            // request as a documented zero-pair sentinel keeps integration code
            // (which may probe spurious coordinate pairs while building joint
            // Hessian sparsity patterns) panic-free.
            if psi_i >= psi_penalty_cache.len() || psi_j >= psi_penalty_cache.len() {
                return HyperCoordPair::zero();
            }
            let cache_i = &psi_penalty_cache[psi_i];
            let cache_j = &psi_penalty_cache[psi_j];

            // Get family-provided second-order likelihood terms.
            let psi2 = if let Some(workspace) = psi_workspace.as_ref() {
                workspace.second_order_terms(psi_i, psi_j).ok().flatten()
            } else {
                family_arc
                    .exact_newton_joint_psisecond_order_terms(
                        &synced_arc,
                        &specs_arc,
                        &derivative_blocks,
                        psi_i,
                        psi_j,
                    )
                    .ok()
                    .flatten()
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

            // Assemble S_{ψ_i ψ_j} only on the touched block.
            let ld_s = if cache_i.block_idx == cache_j.block_idx {
                let p_block = cache_i.end - cache_i.start;
                let deriv_i = &derivative_blocks[cache_i.block_idx][cache_i.local_idx];
                let s_local = assemble_block_local_s_psi_psi(
                    deriv_i,
                    cache_j.local_idx,
                    &per_block[cache_i.block_idx],
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
                    let block_drift: Arc<dyn HyperOperator> =
                        Arc::new(crate::solver::estimate::reml::unified::BlockLocalDrift {
                            local: s_local.clone(),
                            start: cache_i.start,
                            end: cache_i.end,
                            total_dim: total,
                        });
                    b_operator = Some(match b_operator.take() {
                        Some(existing) => {
                            let existing_arc: Arc<dyn HyperOperator> = Arc::from(existing);
                            Box::new(
                                crate::solver::estimate::reml::unified::CompositeHyperOperator {
                                    dense: None,
                                    operators: vec![existing_arc, block_drift],
                                    dim_hint: total,
                                },
                            ) as Box<dyn HyperOperator>
                        }
                        None => Box::new(crate::solver::estimate::reml::unified::BlockLocalDrift {
                            local: s_local.clone(),
                            start: cache_i.start,
                            end: cache_i.end,
                            total_dim: total,
                        }) as Box<dyn HyperOperator>,
                    });
                }

                if let Some(ref logdet_blocks) = *s_logdet_block_cache {
                    let pld = &logdet_blocks[cache_i.block_idx];
                    let s_psi_i = cache_i
                        .s_local
                        .as_ref()
                        .expect("psi cache should include S_psi when penalty logdet is active");
                    let s_psi_j = cache_j
                        .s_local
                        .as_ref()
                        .expect("psi cache should include S_psi when penalty logdet is active");
                    // τ-Hessian: tr(S⁺ S_{ψi ψj}) − tr(S⁺ S_ψi S⁺ S_ψj) + 2 tr(Σ₊⁻² L_i L_j^T)
                    pld.tau_hessian_component(s_psi_i, s_psi_j, Some(&s_local))
                } else {
                    0.0
                }
            } else {
                0.0
            };

            HyperCoordPair {
                a,
                g,
                b_mat,
                b_operator,
                ld_s,
            }
        }) as Box<dyn Fn(usize, usize) -> HyperCoordPair + Send + Sync>
    };

    // ρ-ψ pair callback
    let rho_ext = {
        let per_block = Arc::clone(&per_block);
        let derivative_blocks = Arc::clone(&derivative_blocks);
        let beta_arc = Arc::clone(&beta_arc);
        let psi_penalty_cache = Arc::clone(&psi_penalty_cache);
        let rho_penalty_cache = Arc::clone(&rho_penalty_cache);
        let s_logdet_block_cache = Arc::clone(&s_logdet_block_cache);

        Box::new(move |rho_k: usize, psi_j: usize| -> HyperCoordPair {
            if rho_k >= rho_penalty_cache.len() || psi_j >= psi_penalty_cache.len() {
                return HyperCoordPair::zero();
            }
            let rho_cache = &rho_penalty_cache[rho_k];
            let psi_cache = &psi_penalty_cache[psi_j];
            let mut a = 0.0;
            let mut g = Array1::<f64>::zeros(total);
            let mut b_mat = Array2::<f64>::zeros((total, total));

            // S_{ρ_k, ψ_j} = λ_k ∂S_k/∂ψ_j.
            // Only nonzero when both coordinates share the same block and the
            // ψ derivative touches the k-th penalty.
            let ld_s = if rho_cache.block_idx == psi_cache.block_idx {
                let p_block = rho_cache.end - rho_cache.start;
                let deriv = &derivative_blocks[psi_cache.block_idx][psi_cache.local_idx];
                let lambda_k = per_block[rho_cache.block_idx][rho_cache.penalty_idx].exp();
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
                        .expect("psi cache should include S_psi when penalty logdet is active");
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

            HyperCoordPair {
                a,
                g,
                b_mat,
                b_operator: None,
                ld_s,
            }
        }) as Box<dyn Fn(usize, usize) -> HyperCoordPair + Send + Sync>
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
    derivative_blocks_arc: SharedDerivativeBlocks,
    hessian_beta_independent: bool,
    psi_workspace: Option<Arc<dyn ExactNewtonJointPsiWorkspace>>,
) -> Option<FixedDriftDerivFn> {
    if hessian_beta_independent {
        // Likelihood Hessian is β-independent; M_i ≡ 0.
        return None;
    }

    let synced_arc = Arc::new(synced_states.to_vec());
    let specs_arc = Arc::new(specs.to_vec());
    let family_arc = Arc::new(family.clone());
    let psi_workspace = psi_workspace;

    Some(Box::new(
        move |ext_idx: usize, direction: &Array1<f64>| -> Option<DriftDerivResult> {
            // The family hook takes a psi index (0-based within ψ coordinates)
            // and a flattened coefficient direction.
            if let Some(workspace) = psi_workspace.as_ref() {
                workspace
                    .hessian_directional_derivative(ext_idx, direction)
                    .ok()
                    .flatten()
            } else {
                family_arc
                    .exact_newton_joint_psihessian_directional_derivative(
                        &synced_arc,
                        &specs_arc,
                        &derivative_blocks_arc,
                        ext_idx,
                        direction,
                    )
                    .ok()
                    .flatten()
                    .map(DriftDerivResult::Dense)
            }
        },
    ))
}


fn evaluate_custom_family_hyper_internal<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
    penalty_counts: &[usize],
    rho_current: &Array1<f64>,
    derivative_blocks: &[Vec<CustomFamilyBlockPsiDerivative>],
    warm_start: Option<&ConstrainedWarmStart>,
    rho_prior: crate::types::RhoPrior,
    eval_mode: EvalMode,
) -> Result<OuterObjectiveEvalResult, CustomFamilyError> {
    evaluate_custom_family_hyper_internal_shared(
        family,
        specs,
        options,
        penalty_counts,
        rho_current,
        Arc::new(derivative_blocks.to_vec()),
        warm_start,
        rho_prior,
        eval_mode,
    )
}


fn evaluate_custom_family_hyper_internal_shared<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
    penalty_counts: &[usize],
    rho_current: &Array1<f64>,
    derivative_blocks: SharedDerivativeBlocks,
    warm_start: Option<&ConstrainedWarmStart>,
    rho_prior: crate::types::RhoPrior,
    eval_mode: EvalMode,
) -> Result<OuterObjectiveEvalResult, CustomFamilyError> {
    if derivative_blocks.len() != specs.len() {
        crate::bail_dim_custom!(
            "joint hyper derivative block count mismatch: got {}, expected {}",
            derivative_blocks.len(),
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
    let psi_dim = derivative_blocks.iter().map(Vec::len).sum::<usize>();
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
    let mut inner = inner_blockwise_fit(
        family,
        specs,
        &per_block,
        options,
        psi_safe_warm_start.as_ref().or(warm_start),
    )?;
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
    let moderidge = if options.ridge_policy.include_quadratic_penalty {
        ridge
    } else {
        0.0
    };
    let extra_logdet_ridge = if options.ridge_policy.include_penalty_logdet
        && !options.ridge_policy.include_quadratic_penalty
    {
        ridge
    } else {
        0.0
    };

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
        // par_iter — see `warm_up_outer_caches` doc.
        if let Some(workspace) = hessian_workspace.as_ref() {
            workspace.warm_up_outer_caches()?;
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
                    let lambdas = per_block[b].mapv(f64::exp);
                    let mut s_lambda = Array2::<f64>::zeros((p, p));
                    for (k, s) in spec.penalties.iter().enumerate() {
                        s.add_scaled_to(lambdas[k], &mut s_lambda);
                    }
                    let ridge_hint = if options.ridge_policy.include_penalty_logdet {
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

        // Build ψ HyperCoords, pair callbacks, and drift derivative callback.
        let hessian_beta_independent = !family.exact_newton_joint_hessian_beta_dependent();
        let psi_workspace = if eval_mode != EvalMode::ValueOnly
            && (eval_mode == EvalMode::ValueGradientHessian
                || family.exact_newton_joint_psi_workspace_for_first_order_terms())
        {
            family.exact_newton_joint_psi_workspace_with_options(
                synced_joint_states.as_ref(),
                specs,
                derivative_blocks.as_ref(),
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
                derivative_blocks.as_ref(),
                &beta_flat,
                rho_slice,
                penalty_counts,
                s_logdet_blocks.as_deref(),
                hessian_beta_independent,
                psi_workspace.clone(),
            )?;

            let (ext_ext_fn, rho_ext_fn, drift_fn, contracted_psi_fn) =
                if eval_mode == EvalMode::ValueGradientHessian {
                    let (ext_ext_fn, rho_ext_fn) = build_psi_pair_callbacks(
                        family,
                        synced_joint_states.as_ref(),
                        specs,
                        Arc::clone(&derivative_blocks),
                        &beta_flat,
                        rho_slice,
                        penalty_counts,
                        s_logdet_blocks.as_deref(),
                        psi_workspace.clone(),
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
                        Arc::clone(&derivative_blocks),
                        &beta_flat,
                        rho_slice,
                        penalty_counts,
                        s_logdet_blocks.as_deref(),
                        psi_workspace.clone(),
                    )?;
                    let drift_fn = build_psi_drift_deriv_callback(
                        family,
                        synced_joint_states.as_ref(),
                        specs,
                        Arc::clone(&derivative_blocks),
                        hessian_beta_independent,
                        psi_workspace,
                    );
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
            // ψ-bearing path (matern/duchon marginal-slope kernel length-scales):
            // use the projected #752 generalized determinant for value AND
            // gradient AND Hessian — all produced by this single call, so they are
            // consistent by construction. This is the route the clustered-PC
            // matern bernoulli/survival marginal-slope fits take, where the
            // range(Sλ)-only determinant dropped the penalty-null trend likelihood
            // determinant and froze the outer gradient (gam#808/#787). No batched
            // override is possible here (it is gated to psi_dim==0).
            family.use_projected_penalty_logdet(),
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
                derivative_blocks.as_ref(),
                rho_current,
                hessian_workspace.clone(),
                eval_mode,
            )?,
            custom_family_outer_jeffreys_hphi(family, &inner.block_states, specs, &ranges)?,
            custom_family_outer_jeffreys_hphi_drift(family, &inner.block_states, specs, &ranges)?,
        )?;

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
    let has_configured_rho_prior = !matches!(rho_prior, crate::types::RhoPrior::Flat);
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
        let derivative_blocks_for_batch =
            vec![Vec::<CustomFamilyBlockPsiDerivative>::new(); specs.len()];
        if let Ok(Some(batch)) = family.batched_outer_gradient_terms(
            &synced_states_for_batch,
            specs,
            &derivative_blocks_for_batch,
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
                        crate::types::RhoPrior::Flat,
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
                        outer_hessian: crate::solver::outer_strategy::HessianResult::Unavailable,
                        warm_start: value_only.warm_start,
                        inner_converged: inner.converged,
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
                derivative_blocks.as_ref(),
                rho_current,
                inner.joint_workspace.clone(),
                eval_mode,
            )?,
            custom_family_outer_jeffreys_hphi(family, &inner.block_states, specs, &ranges)?,
            custom_family_outer_jeffreys_hphi_drift(family, &inner.block_states, specs, &ranges)?,
        )?;

        let mut eval_result = eval_result;
        if let Some(batched_grad) = batched_gradient_override.take()
            && batched_grad.len() == eval_result.gradient.len()
        {
            eval_result.gradient = batched_grad;
        }
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
            let w = floor_positiveworking_weights(working_weights, options.minweight);
            let (xtwx, _) = weighted_normal_equations(x_dyn, &w, None)?;
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
                let wwork = floor_positiveworking_weights(working_weights, options.minweight);
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

    let eval_result = joint_outer_evaluate(
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
            derivative_blocks.as_ref(),
            rho_current,
            inner.joint_workspace.clone(),
            eval_mode,
        )?,
        robust_jeffreys_hphi,
        custom_family_outer_jeffreys_hphi_drift(family, &inner.block_states, specs, &ranges)?,
    )?;

    Ok(eval_result)
}


pub fn evaluate_custom_family_joint_hyper<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
    rho_current: &Array1<f64>,
    derivative_blocks: &[Vec<CustomFamilyBlockPsiDerivative>],
    warm_start: Option<&CustomFamilyWarmStart>,
    eval_mode: EvalMode,
) -> Result<CustomFamilyJointHyperResult, CustomFamilyError> {
    let penalty_counts = validate_blockspecs(specs)?;
    let has_psi_derivatives = derivative_blocks.iter().any(|block| !block.is_empty());
    let (eval_options, strict_warm_start) =
        derivative_quality_options_and_warm_start(options, warm_start, has_psi_derivatives);
    let eval_result = evaluate_custom_family_hyper_internal(
        family,
        specs,
        &eval_options,
        &penalty_counts,
        rho_current,
        derivative_blocks,
        strict_warm_start
            .as_ref()
            .map(|w| &w.inner)
            .or_else(|| warm_start.map(|w| &w.inner)),
        crate::types::RhoPrior::Flat,
        eval_mode,
    )?;
    Ok(outer_eval_result_to_joint_hyper_result(eval_result))
}


pub(crate) fn evaluate_custom_family_joint_hyper_shared<
    F: CustomFamily + Clone + Send + Sync + 'static,
>(
    family: &F,
    specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
    rho_current: &Array1<f64>,
    derivative_blocks: SharedDerivativeBlocks,
    warm_start: Option<&CustomFamilyWarmStart>,
    eval_mode: EvalMode,
) -> Result<CustomFamilyJointHyperResult, CustomFamilyError> {
    let penalty_counts = validate_blockspecs(specs)?;
    let has_psi_derivatives = derivative_blocks.iter().any(|block| !block.is_empty());
    let (eval_options, strict_warm_start) =
        derivative_quality_options_and_warm_start(options, warm_start, has_psi_derivatives);
    let eval_result = evaluate_custom_family_hyper_internal_shared(
        family,
        specs,
        &eval_options,
        &penalty_counts,
        rho_current,
        derivative_blocks,
        strict_warm_start
            .as_ref()
            .map(|w| &w.inner)
            .or_else(|| warm_start.map(|w| &w.inner)),
        crate::types::RhoPrior::Flat,
        eval_mode,
    )?;
    Ok(outer_eval_result_to_joint_hyper_result(eval_result))
}


fn derivative_quality_options_and_warm_start(
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
    let direct_joint_hyper_inner_tol = eval_options
        .outer_tol
        .max(DIRECT_JOINT_HYPER_INNER_TOL_FLOOR);
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


pub(crate) fn joint_hyper_options_for_outer_tolerance(
    options: &BlockwiseFitOptions,
    outer_tol: f64,
) -> BlockwiseFitOptions {
    let mut eval_options = options.clone();
    eval_options.outer_tol = eval_options.outer_tol.max(outer_tol);
    eval_options
}


fn evaluate_custom_family_joint_hyper_efs_internal_shared<
    F: CustomFamily + Clone + Send + Sync + 'static,
>(
    family: &F,
    specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
    penalty_counts: &[usize],
    rho_current: &Array1<f64>,
    derivative_blocks: SharedDerivativeBlocks,
    warm_start: Option<&ConstrainedWarmStart>,
) -> Result<
    (
        crate::solver::outer_strategy::EfsEval,
        ConstrainedWarmStart,
        bool,
    ),
    CustomFamilyError,
> {
    if derivative_blocks.len() != specs.len() {
        crate::bail_dim_custom!(
            "joint hyper derivative block count mismatch: got {}, expected {}",
            derivative_blocks.len(),
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
    let psi_dim = derivative_blocks.iter().map(Vec::len).sum::<usize>();
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
        return nonconverged_outer_efs_result(
            &inner,
            rho_current,
            theta_dim,
            include_logdet_h,
            include_logdet_s,
            "custom-family joint-hyper EFS non-converged inner solve",
        )
        .map_err(CustomFamilyError::from);
    }
    let ridge = effective_solverridge(options.ridge_floor);
    let moderidge = if options.ridge_policy.include_quadratic_penalty {
        ridge
    } else {
        0.0
    };
    let extra_logdet_ridge = if options.ridge_policy.include_penalty_logdet
        && !options.ridge_policy.include_quadratic_penalty
    {
        ridge
    } else {
        0.0
    };

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
    // par_iter — see `warm_up_outer_caches` doc.
    if let Some(workspace) = hessian_workspace.as_ref() {
        workspace.warm_up_outer_caches()?;
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
                let lambdas = per_block[b].mapv(f64::exp);
                let mut s_lambda = Array2::<f64>::zeros((p, p));
                for (k, s) in spec.penalties.iter().enumerate() {
                    s.add_scaled_to(lambdas[k], &mut s_lambda);
                }
                let ridge_hint = if options.ridge_policy.include_penalty_logdet {
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
            derivative_blocks.as_ref(),
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
        derivative_blocks.as_ref(),
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
        crate::types::RhoPrior::Flat,
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

    Ok((efs_eval, warm, inner.converged))
}


/// Evaluate the joint custom-family hyper-surface in fixed-point form for the
/// outer EFS / hybrid-EFS planners.
pub fn evaluate_custom_family_joint_hyper_efs<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
    rho_current: &Array1<f64>,
    derivative_blocks: &[Vec<CustomFamilyBlockPsiDerivative>],
    warm_start: Option<&CustomFamilyWarmStart>,
) -> Result<CustomFamilyJointHyperEfsResult, CustomFamilyError> {
    // Borrowed entry point: lift the `&[Vec<…>]` derivative blocks into a
    // `SharedDerivativeBlocks` (`Arc<Vec<Vec<…>>>`) and delegate to the single
    // source of truth. All validation, the empty-block fast path, and the
    // internal evaluator dispatch live in `…_efs_shared`.
    evaluate_custom_family_joint_hyper_efs_shared(
        family,
        specs,
        options,
        rho_current,
        Arc::new(derivative_blocks.to_vec()),
        warm_start,
    )
}


pub(crate) fn evaluate_custom_family_joint_hyper_efs_shared<
    F: CustomFamily + Clone + Send + Sync + 'static,
>(
    family: &F,
    specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
    rho_current: &Array1<f64>,
    derivative_blocks: SharedDerivativeBlocks,
    warm_start: Option<&CustomFamilyWarmStart>,
) -> Result<CustomFamilyJointHyperEfsResult, CustomFamilyError> {
    let penalty_counts = validate_blockspecs(specs)?;
    if derivative_blocks.len() != specs.len() {
        crate::bail_dim_custom!(
            "joint hyper derivative block count mismatch: got {}, expected {}",
            derivative_blocks.len(),
            specs.len()
        );
    }
    let (efs_eval, warm_start, inner_converged) = if derivative_blocks.iter().all(Vec::is_empty) {
        outerobjectiveefs(
            family,
            specs,
            options,
            &penalty_counts,
            rho_current,
            warm_start.map(|w| &w.inner),
            crate::types::RhoPrior::Flat,
        )
        .map_err(CustomFamilyError::from)?
    } else {
        evaluate_custom_family_joint_hyper_efs_internal_shared(
            family,
            specs,
            options,
            &penalty_counts,
            rho_current,
            derivative_blocks,
            warm_start.map(|w| &w.inner),
        )?
    };
    Ok(outer_efs_result_to_joint_hyper_efs_result(
        efs_eval,
        warm_start,
        inner_converged,
    ))
}


fn block_param_ranges(specs: &[ParameterBlockSpec]) -> Vec<(usize, usize)> {
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
fn build_joint_jeffreys_subspace(
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
        let subspace = crate::estimate::reml::jeffreys_subspace::jeffreys_subspace_from_penalty(
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
fn jeffreys_term_skippable_for_source(
    source: &JointHessianSource,
    total_p: usize,
) -> Result<bool, String> {
    // Below the dense-eigh-is-cheap threshold the inner `jeffreys_term_skippable_via_matvec`
    // short-circuits to `false` anyway; bail early so small fits (e.g. BMS p≈51)
    // pay nothing for the pre-check and run the exact dense path unchanged.
    if total_p < crate::estimate::reml::jeffreys_subspace::CHEAP_CONDITIONING_PRECHECK_MIN_DIM {
        return Ok(false);
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
    crate::estimate::reml::jeffreys_subspace::jeffreys_term_skippable_via_matvec(hv, total_p)
}


/// Evaluate ONLY the Jeffreys objective value `Phi = 1/2 log|Z_J^T H Z_J|` at
/// the current working point. Cheaper than the full term (no directional
/// derivatives), used to keep the trust-region accept/reject objective
/// consistent with the Jeffreys-modified Newton step. Returns `0.0` when there
/// is no coefficient system, the family exposes no exact joint Hessian,
/// or the reduced Fisher information is not yet SPD (the value contribution is
/// then simply omitted for that trial point — the step machinery still bounds
/// the coefficient, and the next accepted cycle re-folds a finite value).
fn custom_family_joint_jeffreys_value<F: CustomFamily + Clone + Send + Sync + 'static>(
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
    match crate::estimate::reml::jeffreys_subspace::joint_jeffreys_term(
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
fn custom_family_joint_jeffreys_term<F: CustomFamily + Clone + Send + Sync + 'static>(
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
    let term = crate::estimate::reml::jeffreys_subspace::joint_jeffreys_term(
        h_joint.view(),
        z_joint.view(),
        |direction: &Array1<f64>| {
            family.joint_jeffreys_information_directional_derivative_with_specs(
                states, specs, direction,
            )
        },
    )?;
    Ok(Some(term))
}


const JEFFREYS_REDUCED_INFO_RELATIVE_FLOOR: f64 = 1e-10;

const JEFFREYS_REDUCED_INFO_ABSOLUTE_FLOOR: f64 = 1e-12;

const JEFFREYS_CONDITIONING_GATE_RELATIVE: f64 = 1e-8;

const JEFFREYS_CONDITIONING_GATE_ABSOLUTE: f64 = 1.0;

const JEFFREYS_CONDITIONING_GATE_ABSOLUTE_CLEAR: f64 = 16.0;

const JEFFREYS_CONDITIONING_GATE_RELATIVE_CLEAR: f64 = 1e-6;


#[inline]
fn custom_family_jeffreys_cap(floor: f64) -> f64 {
    JEFFREYS_CONDITIONING_GATE_ABSOLUTE_CLEAR.max(floor)
}


#[inline]
fn custom_family_jeffreys_floored_inverse(lam: f64, floor: f64) -> f64 {
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
fn custom_family_jeffreys_conditioning_gate_weight(lambda_min: f64, lambda_max: f64) -> f64 {
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


fn custom_family_joint_jeffreys_contract_weight(
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


fn custom_family_joint_jeffreys_second_order_completion<
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
            crate::estimate::reml::jeffreys_subspace::joint_jeffreys_second_order_completion(
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
/// CORRECTNESS NOTE (was a bug — see `custom_family_outer_jeffreys_hphi_drift`).
/// `H_Φ` has no EXPLICIT ρ-dependence, but it DOES depend on ρ implicitly through
/// the mode β̂(ρ): `H_Φ = H_Φ(β̂(ρ))` because it is built from `H_id = Z_Jᵀ H Z_J`
/// and `D_a = Z_Jᵀ ∂_a H Z_J`, both functions of β̂. So the exact outer gradient
/// of `½ log|H+S_λ+H_Φ|` carries a `½ tr[(·)⁻¹ D_β H_Φ[v_k]]` drift term ALONGSIDE
/// the likelihood drift `D_β H[v_k]`. Folding `H_Φ` into the `HessianOperator`
/// (the `(·)⁻¹` kernel and `logdet()`) is necessary but NOT sufficient: the
/// trace contraction must ALSO include `D_β H_Φ[v_k]`, supplied by the companion
/// drift wrapper. Without it the analytic gradient describes a DIFFERENT objective
/// than the value, breaking the line search / KKT certification exactly in the
/// near-separating regime where the Jeffreys term is active.
fn custom_family_outer_jeffreys_hphi<F: CustomFamily + Clone + Send + Sync + 'static>(
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
    // The contracted trace hook may supply it at any width; the pairwise
    // `p(p+1)/2` fallback stays capped. `None` degrades safely to the
    // divided-difference solve.
    let total_p = ranges.last().map(|(_, e)| *e).unwrap_or(0);
    let mut completion: Option<Array2<f64>> = None;
    let completion_pairwise_fallback = total_p <= JEFFREYS_COMPLETION_MAX_P;
    let completion_requested = completion_pairwise_fallback
        || family.joint_jeffreys_information_contracted_trace_hessian_available();
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
            completion_pairwise_fallback,
        )?;
    }
    Ok(Some((phi, hphi, completion)))
}


fn batched_outer_gradient_contract_allows_override(
    robust_jeffreys_hphi: Option<&Array2<f64>>,
) -> bool {
    match robust_jeffreys_hphi {
        None => true,
        Some(hphi) => hphi.iter().all(|value| *value == 0.0),
    }
}


/// Build the Tier-B Jeffreys-curvature drift closure `D_β H_Φ[δβ]` for the outer
/// gradient, evaluated at the current outer point (states = β̂(ρ)).
///
/// THE FIX. The outer LAML objective folds `H_Φ` into `½ log|H + S_λ + H_Φ|`;
/// because `H_Φ` depends on ρ through β̂, the exact gradient's trace contraction
/// must include `½ tr[(H+S_λ+H_Φ)⁻¹ D_β H_Φ[v_k]]`. The released Tier-B path
/// supplied ONLY the likelihood-Hessian drift `D_β H[v_k]`, so the analytic
/// gradient omitted `H_Φ`'s mode-response drift — wrong precisely when Jeffreys
/// is active. This returns the missing drift as a `Send + Sync + 'static` closure
/// the `JeffreysHphiAwareJointDerivatives` wrapper folds into the first-order
/// trace, mirroring Tier-A's `FirthAwareGlmDerivatives` `−D(Hφ)[B_k]` term.
///
/// The closure takes the mode-response direction `δβ = dβ̂/dρ_k` (the wrapper
/// performs `v_k → δβ = −v_k`) and returns `D_β H_Φ[δβ]`. Returns `None` when
/// there is no coefficient system — i.e. exactly when
/// `custom_family_outer_jeffreys_hphi` itself returns `None`. The per-direction
/// conditioning gate and floored
/// pseudo-inverse inside `joint_jeffreys_hphi_directional_derivative` reproduce
/// the value path's, so when the value's `H_Φ` is zero (gated/clean fit) the
/// drift is identically zero too.
fn custom_family_outer_jeffreys_hphi_drift<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    states: &[ParameterBlockState],
    specs: &[ParameterBlockSpec],
    ranges: &[(usize, usize)],
) -> Result<Option<JeffreysHphiDriftFn>, String> {
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
    // Snapshot the joint Hessian H(β̂) at the current outer point. If the family
    // exposes no exact joint Hessian the Jeffreys term is inapplicable (matching
    // `custom_family_joint_jeffreys_term`), so no drift is installed.
    let h_joint = match family.joint_jeffreys_information_with_specs(states, specs)? {
        Some(h) => h,
        None => return Ok(None),
    };
    if h_joint.nrows() != total_p || h_joint.ncols() != total_p {
        return Ok(None);
    }
    // Own everything the closure needs so it is `'static + Send + Sync`. β̂ is
    // fixed across the single outer evaluation, so capturing the snapshot states
    // is correct; the closure recomputes the exact directional derivatives of the
    // joint Hessian at that point for each mode-response direction.
    let family_owned = family.clone();
    let states_owned: Vec<ParameterBlockState> = states.to_vec();
    let specs_owned: Vec<ParameterBlockSpec> = specs.to_vec();
    let z_columns = z_joint.clone();
    let drift: JeffreysHphiDriftFn = Arc::new(move |delta: &Array1<f64>| {
        crate::estimate::reml::jeffreys_subspace::joint_jeffreys_hphi_directional_derivative(
            h_joint.view(),
            z_columns.view(),
            delta,
            |direction: &Array1<f64>| {
                family_owned.joint_jeffreys_information_directional_derivative_with_specs(
                    &states_owned,
                    &specs_owned,
                    direction,
                )
            },
            |u: &Array1<f64>, v: &Array1<f64>| {
                family_owned.joint_jeffreys_information_second_directional_derivative_with_specs(
                    &states_owned,
                    &specs_owned,
                    u,
                    v,
                )
            },
        )
        .map(Some)
    });
    Ok(Some(drift))
}


const JOINT_MATRIX_FREE_MIN_DIM: usize = 512;

const JOINT_MATRIX_FREE_MIN_ROWS: usize = 50_000;

const JOINT_MATRIX_FREE_MIN_DIM_AT_LARGE_N: usize = 128;

const JOINT_MATRIX_FREE_MIN_LINEAR_WORK: usize = 4_000_000;

const JOINT_TRACE_STABILITY_RIDGE: f64 = 1e-10;

const JOINT_PCG_MAX_ITER_MULTIPLIER: usize = 4;


pub(crate) fn joint_exact_analytic_outer_hessian_available() -> bool {
    true
}


fn joint_observation_count(states: &[ParameterBlockState]) -> usize {
    states
        .iter()
        .map(|state| state.eta.len())
        .max()
        .unwrap_or(0)
}


/// Whether the unified evaluator will pick the matrix-free joint Hessian path
/// for a problem of size `(total_p, total_n)`. Exposed at crate scope so
/// families with matrix-free operators can branch their `coefficient_hessian_cost`
/// estimate on the same predicate the evaluator will use at fit time.
///
/// For large-scale row counts with only tens of coefficients, exact
/// materialization is bounded by `total_p` Hessian-vector products and then a
/// tiny dense factorization. That is cheaper and more predictable than PCG when
/// each matrix-free product streams all rows through expensive FLEX marginal-
/// slope kernels and the initial joint Hessian is ill-conditioned. Keep the
/// matrix-free route for genuinely wide joint systems, where `total_p` dense
/// products and factorization dominate.
pub(crate) fn use_joint_matrix_free_path(total_p: usize, total_n: usize) -> bool {
    total_p >= JOINT_MATRIX_FREE_MIN_DIM
        || (total_n >= JOINT_MATRIX_FREE_MIN_ROWS
            && total_p >= JOINT_MATRIX_FREE_MIN_DIM_AT_LARGE_N)
        || (total_p >= JOINT_MATRIX_FREE_MIN_DIM_AT_LARGE_N
            && total_n.saturating_mul(total_p) >= JOINT_MATRIX_FREE_MIN_LINEAR_WORK)
}


fn apply_joint_block_penalty(
    ranges: &[(usize, usize)],
    s_lambdas: &[Array2<f64>],
    vector: &Array1<f64>,
    diagonal_ridge: f64,
    joint_full_width: Option<&crate::families::joint_penalty::JointPenaltyBundle>,
) -> Array1<f64> {
    let mut out = Array1::<f64>::zeros(vector.len());
    apply_joint_block_penalty_into(
        ranges,
        s_lambdas,
        vector,
        diagonal_ridge,
        &mut out,
        joint_full_width,
    );
    out
}


/// In-place variant of [`apply_joint_block_penalty`]. Caller supplies the
/// output buffer to eliminate per-call allocation.
///
/// Uses `fast_av_view_into` to write directly into the per-block slice of
/// `out`, avoiding the per-block intermediate `Array1` from `fast_av`. At
/// large scale this is invoked inside the PCG matvec closure (called
/// once per CG iter, hundreds-to-thousands of times per outer iter per
/// the perf-scout report).
fn apply_joint_block_penalty_into(
    ranges: &[(usize, usize)],
    s_lambdas: &[Array2<f64>],
    vector: &Array1<f64>,
    diagonal_ridge: f64,
    out: &mut Array1<f64>,
    joint_full_width: Option<&crate::families::joint_penalty::JointPenaltyBundle>,
) {
    assert_eq!(out.len(), vector.len());
    assert!(s_lambdas.len() <= ranges.len());
    out.fill(0.0);

    if s_lambdas.len() <= 1 {
        for (b, s_lambda) in s_lambdas.iter().enumerate() {
            let (start, end) = ranges[b];
            let block = vector.slice(s![start..end]);
            let mut out_slice = out.slice_mut(s![start..end]);
            crate::linalg::faer_ndarray::fast_av_view_into(s_lambda, &block, out_slice.view_mut());
        }
        if diagonal_ridge > 0.0 {
            out.scaled_add(diagonal_ridge, vector);
        }
        if let Some(bundle) = joint_full_width
            && !bundle.is_empty()
        {
            bundle.add_apply_into(vector.view(), out);
        }
        return;
    }

    if out.as_slice_mut().is_none() {
        for (b, s_lambda) in s_lambdas.iter().enumerate() {
            let (start, end) = ranges[b];
            let block = vector.slice(s![start..end]);
            let mut out_slice = out.slice_mut(s![start..end]);
            crate::linalg::faer_ndarray::fast_av_view_into(s_lambda, &block, out_slice.view_mut());
        }
        if diagonal_ridge > 0.0 {
            out.scaled_add(diagonal_ridge, vector);
        }
        if let Some(bundle) = joint_full_width
            && !bundle.is_empty()
        {
            bundle.add_apply_into(vector.view(), out);
        }
        return;
    }

    {
        let out_values = out
            .as_slice_mut()
            .expect("joint penalty output should be contiguous");
        let mut out_blocks = Vec::with_capacity(s_lambdas.len());
        let mut remaining = out_values;
        let mut cursor = 0usize;
        for &(start, end) in ranges.iter().take(s_lambdas.len()) {
            assert!(start >= cursor);
            assert!(end >= start);
            let (_, after_gap) = remaining.split_at_mut(start - cursor);
            let (out_block, after_block) = after_gap.split_at_mut(end - start);
            out_blocks.push(out_block);
            remaining = after_block;
            cursor = end;
        }

        use rayon::prelude::*;

        out_blocks
            .into_par_iter()
            .enumerate()
            .for_each(|(b, out_block)| {
                let (start, end) = ranges[b];
                let block = vector.slice(s![start..end]);
                let out_view = ArrayViewMut1::from(out_block);
                crate::linalg::faer_ndarray::fast_av_view_into(&s_lambdas[b], &block, out_view);
            });
    }

    if diagonal_ridge > 0.0 {
        if let (Some(out_values), Some(vector_values)) = (out.as_slice_mut(), vector.as_slice()) {
            use rayon::prelude::*;

            out_values
                .par_iter_mut()
                .zip(vector_values.par_iter())
                .for_each(|(out_value, vector_value)| {
                    *out_value += diagonal_ridge * *vector_value;
                });
        } else {
            out.scaled_add(diagonal_ridge, vector);
        }
    }

    if let Some(bundle) = joint_full_width
        && !bundle.is_empty()
    {
        bundle.add_apply_into(vector.view(), out);
    }
}


/// Penalty-aware Jacobi preconditioner used by every matrix-free PCG path
/// in the inner coefficient solve.
///
/// Builds `diag(H) + Σ_k gershgorin(S_k(λ)) + ridge`, clamped at 1e-10, where
/// `gershgorin(S)[i] = Σ_j |S[i,j]|` is the absolute row-sum (Gershgorin
/// radius) of each penalty block. This strictly dominates `diag(S)` for any
/// penalty with off-diagonal mass — the high-order difference / thin-plate
/// smooths (the cubic-Duchon `[mass, tension, stiffness]` triple, orders
/// [1,2,3] in `WigglePenaltyConfig::cubic_triple_operator_default`) are
/// strongly off-diagonal-dominant, so `S[i,i]` alone understates the
/// operator's true row scale by orders of magnitude there.
///
/// Why the row-sum and not just the diagonal: a plain Jacobi (diagonal-only)
/// preconditioner collapses to `diag(S_λ)` exactly in the saturated-softmax
/// regime, where the data Fisher weight `W = diag(p) − ppᵀ → 0` near the
/// simplex boundary and the data part of `diag(H)` vanishes. When the penalty
/// is off-diagonal-dominant, `diag(S_λ)` is a poor spectral match for
/// `H + S_λ`, leaving PCG with a large effective condition number and only
/// geometric (linear) convergence — the multinomial-penguins grind in #715.
/// The Gershgorin row-sum diagonal tracks the operator's per-coordinate scale
/// (`|S| 𝟙` bounds `S`'s action), tightening the preconditioned spectrum and
/// cutting CG iterations sharply in that regime. It is `≥ diag(S)` entrywise
/// for SPD `S`, so it stays strictly positive and SPD: it changes only the
/// PCG trajectory, never the converged Newton step or the KKT certificate
/// (PCG converges to the same `(H + S_λ)⁻¹ rhs` under any SPD preconditioner).
/// Design docs sometimes call this the "triple-operator penalty
/// preconditioner"; in code it is the single, unified preconditioner shared by
/// all PCG callsites.
///
/// Callers in the PIRLS inner Newton PCG path feed the result as the diagonal
/// rescale every CG iteration: PCG applies `M^{-1}` to residuals directly.
/// Do not square-root or trace-normalize these entries, and do not apply a
/// second preconditioner-side rescale to the returned Newton step.
fn positive_joint_diagonal_entry(value: f64) -> f64 {
    if value.is_finite() && value > 1.0e-10 {
        value
    } else {
        1.0e-10
    }
}


fn joint_penalty_preconditioner_diag(
    base_diagonal: &Array1<f64>,
    ranges: &[(usize, usize)],
    s_lambdas: &[Array2<f64>],
    diagonal_ridge: f64,
    joint_full_width: Option<&crate::families::joint_penalty::JointPenaltyBundle>,
) -> Array1<f64> {
    assert!(s_lambdas.len() <= ranges.len());
    let mut diag = base_diagonal.clone();
    for (b, s_lambda) in s_lambdas.iter().enumerate() {
        let (start, end) = ranges[b];
        assert_eq!(s_lambda.nrows(), end - start);
        assert_eq!(s_lambda.ncols(), end - start);
        // Gershgorin radius: the absolute row-sum `Σ_j |S[i,j]|` of the penalty
        // block, not just its diagonal `S[i,i]`. For an off-diagonal-dominant
        // smooth penalty (high-order difference / thin-plate) this tracks the
        // operator's true per-coordinate scale, where `S[i,i]` understates it.
        // For SPD `S` the row-sum is `≥ |S[i,i]| = S[i,i]`, so the result still
        // strictly dominates the plain-diagonal preconditioner and stays SPD.
        for (local_idx, global_idx) in (start..end).enumerate() {
            let row_abs_sum: f64 = s_lambda
                .row(local_idx)
                .iter()
                .map(|value| value.abs())
                .sum();
            diag[global_idx] += row_abs_sum;
        }
    }
    if diagonal_ridge > 0.0 {
        for value in &mut diag {
            *value += diagonal_ridge;
        }
    }
    if let Some(bundle) = joint_full_width
        && !bundle.is_empty()
    {
        bundle.add_diag(&mut diag);
    }
    diag.mapv(positive_joint_diagonal_entry)
}


fn log_joint_pcg_diagnostics(
    cycle: usize,
    total_p: usize,
    total_n: usize,
    preconditioner_diag: &Array1<f64>,
    info: &crate::linalg::utils::PcgSolveInfo,
) {
    let (diag_min, diag_max) = preconditioner_diag.iter().fold(
        (f64::INFINITY, 0.0_f64),
        |(min_value, max_value), &value| {
            if value.is_finite() {
                (min_value.min(value), max_value.max(value))
            } else {
                (min_value, max_value)
            }
        },
    );
    let diag_ratio = if diag_min.is_finite() && diag_min > 0.0 && diag_max.is_finite() {
        Some(diag_max / diag_min)
    } else {
        None
    };
    log::info!(
        "[PIRLS/blockwise joint-Newton/PCG] cycle={} p={} n={} iters={} rel_res={:.3e} res0={:.3e} res_final={:.3e} res_ratio={:.3e} ritz_cond~{} jacobi_diag_ratio~{}",
        cycle,
        total_p,
        total_n,
        info.iterations,
        info.relative_residual_norm,
        info.initial_residual_norm,
        info.final_residual_norm,
        info.residual_reduction,
        info.condition_estimate
            .map(|value| format!("{value:.3e}"))
            .unwrap_or_else(|| "NA".to_string()),
        diag_ratio
            .map(|value| format!("{value:.3e}"))
            .unwrap_or_else(|| "NA".to_string()),
    );
}


fn add_joint_penalty_to_matrix(
    matrix: &mut Array2<f64>,
    ranges: &[(usize, usize)],
    s_lambdas: &[Array2<f64>],
    diagonal_ridge: f64,
    joint_full_width: Option<&crate::families::joint_penalty::JointPenaltyBundle>,
) {
    for (b, s_lambda) in s_lambdas.iter().enumerate() {
        let (start, end) = ranges[b];
        let mut block = matrix.slice_mut(s![start..end, start..end]);
        block += s_lambda;
    }
    if diagonal_ridge > 0.0 {
        for d in 0..matrix.nrows() {
            matrix[[d, d]] += diagonal_ridge;
        }
    }
    if let Some(bundle) = joint_full_width
        && !bundle.is_empty()
    {
        bundle.add_to_matrix(matrix);
    }
}


fn flatten_state_betas(
    states: &[ParameterBlockState],
    specs: &[ParameterBlockSpec],
) -> Array1<f64> {
    let total = specs.iter().map(|s| s.design.ncols()).sum::<usize>();
    let mut beta = Array1::<f64>::zeros(total);
    let ranges = block_param_ranges(specs);
    for (b, (start, end)) in ranges.into_iter().enumerate() {
        beta.slice_mut(ndarray::s![start..end])
            .assign(&states[b].beta);
    }
    beta
}


fn set_states_from_flat_beta(
    states: &mut [ParameterBlockState],
    specs: &[ParameterBlockSpec],
    beta_flat: &Array1<f64>,
) -> Result<(), String> {
    let ranges = block_param_ranges(specs);
    let total = ranges.last().map(|(_, e)| *e).unwrap_or(0);
    if beta_flat.len() != total {
        return Err(CustomFamilyError::DimensionMismatch {
            reason: format!(
                "flat beta length mismatch: got {}, expected {}",
                beta_flat.len(),
                total
            ),
        }
        .into());
    }
    for (b, (start, end)) in ranges.into_iter().enumerate() {
        states[b]
            .beta
            .assign(&beta_flat.slice(ndarray::s![start..end]).to_owned());
    }
    Ok(())
}


fn synchronized_states_from_flat_beta<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    specs: &[ParameterBlockSpec],
    states: &[ParameterBlockState],
    beta_flat: &Array1<f64>,
) -> Result<Vec<ParameterBlockState>, String> {
    let mut synced = states.to_vec();
    set_states_from_flat_beta(&mut synced, specs, beta_flat)?;
    refresh_all_block_etas(family, specs, &mut synced)?;
    Ok(synced)
}


/// Inf-norm of the penalized stationarity residual with valid KKT multipliers
/// projected out at active linear constraints.
///
/// For a linearly constrained convex quadratic with constraints `Aβ ≥ b`,
/// the KKT conditions at β̂ read
///
///   S·β̂ − ∇ℓ(β̂) = A_activeᵀ λ
///   Aβ̂ − b ≥ 0
///   λ ≥ 0
///   λᵢ(Aᵢβ̂ − bᵢ) = 0
///
/// The residual component represented by nonnegative active multipliers is
/// therefore not a convergence defect. This helper removes that normal-cone
/// component before taking the inf-norm. Axis-aligned lower bounds are just a
/// special case; coupled derivative-guard rows must use the same KKT geometry.
///
/// `known_active_rows`, when provided, seeds the working set with the QP
/// solver's authoritative active rows. Trust-region damping and finite
/// precision can leave the committed β with row slacks slightly above the slack
/// tolerance even though the QP identified the row as binding; slack-based
/// detection alone then misses the row and leaves its Lagrange-multiplier mass
/// in the projected residual. Seeding from the QP's active set is exact; the
/// non-negative-multiplier iteration below then removes any seeded row whose
/// least-squares multiplier turns out to be strictly negative, so the union
/// of (QP active) ∪ (slack-detected) never declares false convergence.
fn projected_stationarity_inf_norm(
    residual: &Array1<f64>,
    beta: &Array1<f64>,
    constraints: Option<&LinearInequalityConstraints>,
    known_active_rows: Option<&[usize]>,
) -> f64 {
    assert_eq!(residual.len(), beta.len());
    let raw_inf = residual.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
    let Some(constraints) = constraints else {
        return raw_inf;
    };
    projected_linear_constraint_stationarity_inf_norm(
        residual,
        beta,
        constraints,
        known_active_rows,
    )
    .unwrap_or(raw_inf)
}


fn projected_linear_constraint_stationarity_inf_norm(
    residual: &Array1<f64>,
    beta: &Array1<f64>,
    constraints: &LinearInequalityConstraints,
    known_active_rows: Option<&[usize]>,
) -> Option<f64> {
    let projected = projected_linear_constraint_stationarity_vector(
        residual,
        beta,
        constraints,
        known_active_rows,
    )?;
    let primal_violation = linear_constraint_primal_violation(beta, constraints)?;
    Some(
        projected
            .iter()
            .fold(0.0_f64, |acc, &v| acc.max(v.abs()))
            .max(primal_violation),
    )
}


fn linear_constraint_primal_violation(
    beta: &Array1<f64>,
    constraints: &LinearInequalityConstraints,
) -> Option<f64> {
    if constraints.a.ncols() != beta.len() || constraints.a.nrows() != constraints.b.len() {
        return None;
    }
    let mut primal_violation = 0.0_f64;
    for row in 0..constraints.a.nrows() {
        if constraints.b[row] == f64::NEG_INFINITY {
            continue;
        }
        if !constraints.b[row].is_finite() {
            return None;
        }
        let value = constraints.a.row(row).dot(beta);
        let slack = value - constraints.b[row];
        if !slack.is_finite() {
            return None;
        }
        primal_violation = primal_violation.max((-slack).max(0.0));
    }
    Some(primal_violation)
}


fn projected_linear_constraint_stationarity_vector(
    residual: &Array1<f64>,
    beta: &Array1<f64>,
    constraints: &LinearInequalityConstraints,
    known_active_rows: Option<&[usize]>,
) -> Option<Array1<f64>> {
    let p = beta.len();
    if residual.len() != p
        || constraints.a.ncols() != p
        || constraints.a.nrows() != constraints.b.len()
    {
        return None;
    }
    let n_rows = constraints.a.nrows();
    // Union the slack-detected active rows with the optional QP-supplied
    // hint. Using a boolean membership table preserves a canonical row order
    // (matching the constraint matrix) so the rank-reduction below is
    // deterministic across calls.
    let mut in_active = vec![false; n_rows];
    if let Some(hint) = known_active_rows {
        for &row in hint {
            if row < n_rows && constraints.b[row].is_finite() {
                in_active[row] = true;
            }
        }
    }
    for row in 0..n_rows {
        if constraints.b[row] == f64::NEG_INFINITY {
            continue;
        }
        if !constraints.b[row].is_finite() {
            return None;
        }
        let a_row = constraints.a.row(row);
        let value = a_row.dot(beta);
        let slack = value - constraints.b[row];
        if !slack.is_finite() {
            return None;
        }
        if in_active[row] {
            continue;
        }
        // Active-row inclusion band for the stationarity-residual cone projection.
        // A constraint binding at the constrained optimum carries a Lagrange
        // multiplier whose mass IS the stationarity residual (`r = A_activeᵀ λ`,
        // λ >= 0); to project it out, every genuinely tight row must be a candidate.
        // The constrained QP only reports rows it drove tight during a
        // non-degenerate step, so monotone derivative-guard rows tight at the
        // optimum but never explicitly stepped sit just above the old `1e-6·scale`
        // band, get excluded, and leave the multiplier unresolved — tripping the
        // `active_set_incomplete` refusal on an exactly constrained-stationary
        // iterate (gam#797 survival time block). Widen the band so every near-tight
        // row is a CANDIDATE; over-inclusion is safe because the downstream NNLS
        // (`project_stationarity_residual_on_constraint_cone`) assigns λ = 0 to any
        // candidate carrying no multiplier mass, so a non-binding row cannot
        // spuriously shrink the residual.
        let scale = value.abs().max(constraints.b[row].abs()).max(1.0);
        let active_tol = 1e-3 * scale + 1e-8;
        if slack <= active_tol {
            in_active[row] = true;
        }
    }
    let active_rows: Vec<usize> = (0..n_rows).filter(|&row| in_active[row]).collect();
    if active_rows.is_empty() {
        return Some(residual.clone());
    }

    let mut a_active = Array2::<f64>::zeros((active_rows.len(), p));
    for (pos, &row) in active_rows.iter().enumerate() {
        a_active.row_mut(pos).assign(&constraints.a.row(row));
    }
    project_stationarity_residual_on_constraint_cone(residual, &a_active)
        .map(|(projected, _)| projected)
}


fn exact_newton_joint_stationarity_inf_norm<F: CustomFamily + ?Sized>(
    family: &F,
    specs: &[ParameterBlockSpec],
    eval: &FamilyEvaluation,
    states: &[ParameterBlockState],
    s_lambdas: &[Array2<f64>],
    ridge: f64,
    ridge_policy: RidgePolicy,
    block_active_sets: Option<&[Option<Vec<usize>>]>,
) -> Result<Option<f64>, String> {
    if eval.blockworking_sets.len() != states.len() || states.len() != s_lambdas.len() {
        return Err(CustomFamilyError::DimensionMismatch {
            reason: "exact-newton joint stationarity check: block dimension mismatch".to_string(),
        }
        .into());
    }
    if specs.len() != states.len() {
        return Err(CustomFamilyError::DimensionMismatch {
            reason: "exact-newton joint stationarity check: spec/state count mismatch".to_string(),
        }
        .into());
    }
    if let Some(sets) = block_active_sets
        && sets.len() != states.len()
    {
        return Err(CustomFamilyError::DimensionMismatch { reason: format!(
            "exact-newton joint stationarity check: active-set count mismatch, got {}, expected {}",
            sets.len(),
            states.len()
        ) }.into());
    }

    let block_constraints = collect_block_linear_constraints(family, states, specs)?;
    let mut inf_norm = 0.0_f64;
    for b in 0..states.len() {
        let gradient = match &eval.blockworking_sets[b] {
            // For exact-Newton families the block score is ∇ log L with respect
            // to that block, while the penalized negative objective is
            //
            //   Q(beta, rho) = -log L(beta) + 0.5 beta^T P_mode(rho) beta,
            //
            // where `P_mode` includes the rho-independent stabilization ridge
            // exactly when that ridge participates in the quadratic objective.
            //
            // The coupled first-order condition is therefore
            //
            //   ∇Q = -∇ log L + P beta = 0.
            //
            // So the exact penalized stationarity residual for block b is
            //
            //   r_b = P_mode,b * beta_b - gradient_b.
            //
            // For blocks with simple lower-bound constraints (e.g. I-spline
            // monotone time coefficients, monotone wiggle coefficients) the
            // residual on an active-bound coordinate is the KKT multiplier
            // λ_j ≥ 0 rather than a convergence defect; the projection in
            // `projected_stationarity_inf_norm` drops those entries so the
            // inf-norm measures only the free-set residual that must be
            // driven to zero. Using only coordinate step size or an
            // unprojected norm can declare convergence too early OR fail to
            // ever declare convergence at a constrained optimum.
            BlockWorkingSet::ExactNewton { gradient, .. } => gradient,
            _ => return Ok(None),
        };
        let mut residual = s_lambdas[b].dot(&states[b].beta) - gradient;
        if ridge_policy.include_quadratic_penalty && ridge > 0.0 {
            residual += &states[b].beta.mapv(|v| ridge * v);
        }
        let block_active_hint = block_active_sets
            .and_then(|sets| sets.get(b))
            .and_then(|opt| opt.as_deref());
        let block_inf = projected_stationarity_inf_norm(
            &residual,
            &states[b].beta,
            block_constraints[b].as_ref(),
            block_active_hint,
        );
        inf_norm = inf_norm.max(block_inf);
    }
    Ok(Some(inf_norm))
}


fn exact_newton_joint_gradient_from_eval(
    eval: &FamilyEvaluation,
    specs: &[ParameterBlockSpec],
    states: &[ParameterBlockState],
) -> Result<Option<Array1<f64>>, String> {
    if eval.blockworking_sets.len() != specs.len() {
        return Err(format!(
            "exact-newton joint gradient extraction: family returned {} block working sets, expected {}",
            eval.blockworking_sets.len(),
            specs.len()
        ));
    }
    if states.len() != specs.len() {
        return Err(CustomFamilyError::DimensionMismatch { reason: format!(
            "exact-newton joint gradient extraction: state count {} does not match spec count {}",
            states.len(),
            specs.len()
        ) }.into());
    }
    let total_p = specs.iter().map(|spec| spec.design.ncols()).sum::<usize>();
    let mut gradient = Array1::<f64>::zeros(total_p);
    let mut offset = 0usize;
    for ((spec, work), state) in specs
        .iter()
        .zip(eval.blockworking_sets.iter())
        .zip(states.iter())
    {
        let width = spec.design.ncols();
        match work {
            BlockWorkingSet::ExactNewton {
                gradient: block_gradient,
                ..
            } => {
                if block_gradient.len() != width {
                    return Err(CustomFamilyError::DimensionMismatch { reason: format!(
                        "exact-newton joint gradient extraction: block gradient length mismatch, got {}, expected {}",
                        block_gradient.len(),
                        width
                    ) }.into());
                }
                gradient
                    .slice_mut(ndarray::s![offset..offset + width])
                    .assign(block_gradient);
            }
            BlockWorkingSet::Diagonal {
                working_response,
                working_weights,
            } => {
                // Recover the per-block log-likelihood score from the IRLS
                // working set.  By construction of the IRLS pseudo-response
                //
                //     z_i = η_i + (∂ℓ/∂η_i) / w_i,
                //
                // so the row score is `w_i (z_i − η_i)` and the
                // coefficient-space score is
                //
                //     ∇_β_b log L = X_b^T (w ⊙ (z − η)).
                //
                // Without this branch the joint-Newton path is unable to
                // assemble its RHS for families that emit Diagonal working
                // sets alongside an exact joint Hessian (e.g. Gaussian
                // location-scale): the inner fit returns non-converged, and
                // the outer evaluator falls into the nonconverged-result
                // branch and reports a zero outer gradient.
                let n = working_response.len();
                if working_weights.len() != n || state.eta.len() != n || spec.design.nrows() != n {
                    return Err(CustomFamilyError::DimensionMismatch { reason: format!(
                        "exact-newton joint gradient extraction: diagonal working-set length mismatch (z={}, w={}, η={}, X_rows={})",
                        working_response.len(),
                        working_weights.len(),
                        state.eta.len(),
                        spec.design.nrows()
                    ) }.into());
                }
                let mut weighted = Array1::<f64>::zeros(n);
                for i in 0..n {
                    weighted[i] = working_weights[i] * (working_response[i] - state.eta[i]);
                }
                let block_gradient =
                    <DesignMatrix as LinearOperator>::apply_transpose(&spec.design, &weighted);
                if block_gradient.len() != width {
                    return Err(CustomFamilyError::DimensionMismatch { reason: format!(
                        "exact-newton joint gradient extraction: diagonal block transpose length mismatch, got {}, expected {}",
                        block_gradient.len(),
                        width
                    ) }.into());
                }
                gradient
                    .slice_mut(ndarray::s![offset..offset + width])
                    .assign(&block_gradient);
            }
        }
        offset += width;
    }
    Ok(Some(gradient))
}


fn exact_newton_joint_stationarity_inf_norm_from_gradient(
    gradient: &Array1<f64>,
    states: &[ParameterBlockState],
    specs: &[ParameterBlockSpec],
    s_lambdas: &[Array2<f64>],
    ridge: f64,
    ridge_policy: RidgePolicy,
    block_constraints: &[Option<LinearInequalityConstraints>],
    block_active_sets: Option<&[Option<Vec<usize>>]>,
) -> Result<f64, String> {
    if states.len() != specs.len() || states.len() != s_lambdas.len() {
        return Err(
            "exact-newton joint stationarity check from gradient: block dimension mismatch"
                .to_string(),
        );
    }
    if block_constraints.len() != states.len() {
        return Err(CustomFamilyError::DimensionMismatch { reason: format!(
            "exact-newton joint stationarity check from gradient: constraint count mismatch, got {}, expected {}",
            block_constraints.len(),
            states.len()
        ) }.into());
    }
    if let Some(sets) = block_active_sets
        && sets.len() != states.len()
    {
        return Err(CustomFamilyError::DimensionMismatch { reason: format!(
            "exact-newton joint stationarity check from gradient: active-set count mismatch, got {}, expected {}",
            sets.len(),
            states.len()
        ) }.into());
    }
    let total_p = specs.iter().map(|spec| spec.design.ncols()).sum::<usize>();
    if gradient.len() != total_p {
        return Err(CustomFamilyError::DimensionMismatch { reason: format!(
            "exact-newton joint stationarity check from gradient: joint gradient length mismatch, got {}, expected {}",
            gradient.len(),
            total_p
        ) }.into());
    }

    // Same KKT projection as `exact_newton_joint_stationarity_inf_norm`:
    // multipliers at active lower bounds are not convergence defects, so we
    // measure only the free-set residual. See `projected_stationarity_inf_norm`
    // for the tolerance choice and its parallel with `projected_gradient_norm`
    // in `pirls.rs`.
    //
    // The optional `block_active_sets` arrives from the joint-Newton inner
    // loop's `cached_active_sets` and carries the QP solver's authoritative
    // active rows per block. Threading it through is what makes the
    // stationarity test correctly fire at the constrained optimum: a damped
    // constrained step may commit β with row slacks slightly above the slack
    // tolerance even though the QP identified the rows as binding, and
    // slack-based detection alone then misses the rows and leaves the
    // Lagrange-multiplier mass in the residual.
    let mut inf_norm = 0.0_f64;
    let mut offset = 0usize;
    for b in 0..states.len() {
        let width = specs[b].design.ncols();
        let mut residual =
            s_lambdas[b].dot(&states[b].beta) - gradient.slice(ndarray::s![offset..offset + width]);
        if ridge_policy.include_quadratic_penalty && ridge > 0.0 {
            residual += &states[b].beta.mapv(|v| ridge * v);
        }
        let block_active_hint = block_active_sets
            .and_then(|sets| sets.get(b))
            .and_then(|opt| opt.as_deref());
        let block_inf = projected_stationarity_inf_norm(
            &residual,
            &states[b].beta,
            block_constraints[b].as_ref(),
            block_active_hint,
        );
        inf_norm = inf_norm.max(block_inf);
        offset += width;
    }
    Ok(inf_norm)
}


fn exact_newton_joint_stationarity_vector_from_gradient(
    gradient: &Array1<f64>,
    states: &[ParameterBlockState],
    specs: &[ParameterBlockSpec],
    s_lambdas: &[Array2<f64>],
    ridge: f64,
    ridge_policy: RidgePolicy,
) -> Result<Array1<f64>, String> {
    if states.len() != specs.len() || states.len() != s_lambdas.len() {
        return Err(
            "exact-newton joint stationarity vector from gradient: block dimension mismatch"
                .to_string(),
        );
    }
    let total_p = specs.iter().map(|spec| spec.design.ncols()).sum::<usize>();
    if gradient.len() != total_p {
        return Err(CustomFamilyError::DimensionMismatch { reason: format!(
            "exact-newton joint stationarity vector from gradient: joint gradient length mismatch, got {}, expected {}",
            gradient.len(),
            total_p
        ) }.into());
    }

    let mut residual = Array1::<f64>::zeros(total_p);
    let mut offset = 0usize;
    for b in 0..states.len() {
        let width = specs[b].design.ncols();
        let start = offset;
        let end = offset + width;
        let mut block = s_lambdas[b].dot(&states[b].beta) - gradient.slice(ndarray::s![start..end]);
        if ridge_policy.include_quadratic_penalty && ridge > 0.0 {
            block += &states[b].beta.mapv(|v| ridge * v);
        }
        residual.slice_mut(ndarray::s![start..end]).assign(&block);
        offset = end;
    }
    Ok(residual)
}


fn exact_newton_joint_projected_stationarity_vector_from_gradient(
    gradient: &Array1<f64>,
    states: &[ParameterBlockState],
    specs: &[ParameterBlockSpec],
    s_lambdas: &[Array2<f64>],
    ridge: f64,
    ridge_policy: RidgePolicy,
    block_constraints: &[Option<LinearInequalityConstraints>],
    block_active_sets: Option<&[Option<Vec<usize>>]>,
) -> Result<Array1<f64>, String> {
    if states.len() != specs.len()
        || states.len() != s_lambdas.len()
        || states.len() != block_constraints.len()
    {
        return Err(
            "exact-newton projected stationarity vector from gradient: block dimension mismatch"
                .to_string(),
        );
    }
    if let Some(sets) = block_active_sets
        && sets.len() != states.len()
    {
        return Err(CustomFamilyError::DimensionMismatch { reason: format!(
            "exact-newton projected stationarity vector from gradient: active-set count mismatch, got {}, expected {}",
            sets.len(),
            states.len()
        ) }.into());
    }
    let total_p = specs.iter().map(|spec| spec.design.ncols()).sum::<usize>();
    if gradient.len() != total_p {
        return Err(CustomFamilyError::DimensionMismatch { reason: format!(
            "exact-newton projected stationarity vector from gradient: joint gradient length mismatch, got {}, expected {}",
            gradient.len(),
            total_p
        ) }.into());
    }

    let mut residual = Array1::<f64>::zeros(total_p);
    let mut offset = 0usize;
    for b in 0..states.len() {
        let width = specs[b].design.ncols();
        let start = offset;
        let end = offset + width;
        let mut block = s_lambdas[b].dot(&states[b].beta) - gradient.slice(ndarray::s![start..end]);
        if ridge_policy.include_quadratic_penalty && ridge > 0.0 {
            block += &states[b].beta.mapv(|v| ridge * v);
        }
        if let Some(constraints) = block_constraints[b].as_ref() {
            let block_active_hint = block_active_sets
                .and_then(|sets| sets.get(b))
                .and_then(|opt| opt.as_deref());
            match projected_linear_constraint_stationarity_vector(
                &block,
                &states[b].beta,
                constraints,
                block_active_hint,
            ) {
                Some(projected) => block = projected,
                None => {
                    // Cone projection can only SHRINK the residual (it removes
                    // nonnegative multiplier mass on active rows), so a failed
                    // projection degrades to the conservative unprojected
                    // residual — the convergence test gets harder, never
                    // easier — instead of rejecting the whole seed (#1025:
                    // 'failed to project block 0' killed an otherwise-healthy
                    // competing-risks seed outright).
                    log::warn!(
                        "exact-newton projected stationarity vector: cone projection failed \
                         for block {b}; using the conservative unprojected residual"
                    );
                }
            }
        }
        residual.slice_mut(ndarray::s![start..end]).assign(&block);
        offset = end;
    }
    Ok(residual)
}


/// Build the free-space-projected KKT residual for the IFT correction.
///
/// The active set passed via `block_active_sets` is consumed by the inner
/// projection so the returned vector lies in `range(I − P_normal_cone)`. The
/// [`crate::solver::estimate::reml::unified::ProjectedKktResidual`] return type makes
/// that invariant visible at every call site — callers cannot forget to
/// project, and `reml/unified.rs` cannot accidentally accept an unprojected
/// vector.
fn exact_newton_joint_kkt_residual_for_ift<F: CustomFamily + ?Sized>(
    family: &F,
    specs: &[ParameterBlockSpec],
    states: &[ParameterBlockState],
    s_lambdas: &[Array2<f64>],
    ridge: f64,
    ridge_policy: RidgePolicy,
    block_active_sets: Option<&[Option<Vec<usize>>]>,
) -> Result<Option<ProjectedKktResidual>, String> {
    let eval = family.evaluate(states)?;
    let Some(gradient) = exact_newton_joint_gradient_from_eval(&eval, specs, states)? else {
        return Ok(None);
    };
    let block_constraints = collect_block_linear_constraints(family, states, specs)?;
    exact_newton_joint_projected_kkt_residual_for_ift_from_gradient(
        &gradient,
        specs,
        states,
        s_lambdas,
        ridge,
        ridge_policy,
        &block_constraints,
        block_active_sets,
    )
}


fn exact_newton_joint_kkt_residual_for_ift_from_cached_gradient<F: CustomFamily + ?Sized>(
    family: &F,
    specs: &[ParameterBlockSpec],
    states: &[ParameterBlockState],
    s_lambdas: &[Array2<f64>],
    ridge: f64,
    ridge_policy: RidgePolicy,
    block_active_sets: Option<&[Option<Vec<usize>>]>,
    cached_gradient: Option<&Array1<f64>>,
) -> Result<Option<ProjectedKktResidual>, String> {
    if let Some(gradient) = cached_gradient {
        let block_constraints = collect_block_linear_constraints(family, states, specs)?;
        return exact_newton_joint_projected_kkt_residual_for_ift_from_gradient(
            gradient,
            specs,
            states,
            s_lambdas,
            ridge,
            ridge_policy,
            &block_constraints,
            block_active_sets,
        );
    }
    exact_newton_joint_kkt_residual_for_ift(
        family,
        specs,
        states,
        s_lambdas,
        ridge,
        ridge_policy,
        block_active_sets,
    )
}


fn exact_newton_joint_projected_kkt_residual_for_ift_from_gradient(
    gradient: &Array1<f64>,
    specs: &[ParameterBlockSpec],
    states: &[ParameterBlockState],
    s_lambdas: &[Array2<f64>],
    ridge: f64,
    ridge_policy: RidgePolicy,
    block_constraints: &[Option<LinearInequalityConstraints>],
    block_active_sets: Option<&[Option<Vec<usize>>]>,
) -> Result<Option<ProjectedKktResidual>, String> {
    let residual = exact_newton_joint_projected_stationarity_vector_from_gradient(
        gradient,
        states,
        specs,
        s_lambdas,
        ridge,
        ridge_policy,
        block_constraints,
        block_active_sets,
    )?;
    if residual.iter().all(|v| v.is_finite()) {
        Ok(Some(ProjectedKktResidual::from_active_projected(residual)))
    } else {
        // Surface this clearly: a non-finite projected residual reaches the
        // unified evaluator as `kkt_residual = None`, which then makes the
        // envelope-consistency tripwire fire with "no projected residual"
        // as the suspected cause. Emit the count and magnitude so the
        // failure is diagnosable from a single log line.
        let nan_count = residual.iter().filter(|v| v.is_nan()).count();
        let inf_count = residual.iter().filter(|v| v.is_infinite()).count();
        let finite_max = residual
            .iter()
            .filter(|v| v.is_finite())
            .copied()
            .map(f64::abs)
            .fold(0.0_f64, f64::max);
        log::warn!(
            "[exact-newton kkt-residual projection] dropping projected KKT residual to None: \
             len={} nan_count={} inf_count={} finite_max={:.3e}. The unified evaluator will \
             treat this convergent path as if no residual were available, which silently \
             disables the IFT correction and can trip the envelope-gradient consistency check \
             on near-singular H. Investigate which block produced the non-finite entry.",
            residual.len(),
            nan_count,
            inf_count,
            finite_max,
        );
        Ok(None)
    }
}


fn compute_joint_covariance<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    specs: &[ParameterBlockSpec],
    states: &[ParameterBlockState],
    per_block_log_lambdas: &[Array1<f64>],
    options: &BlockwiseFitOptions,
) -> Result<Array2<f64>, String> {
    let ranges = block_param_ranges(specs);
    let total = ranges.last().map(|(_, e)| *e).unwrap_or(0);
    let Some(mut h) = exact_newton_joint_hessian_symmetrized(
        family,
        states,
        specs,
        total,
        "joint exact-newton Hessian shape mismatch in covariance",
    )?
    else {
        return Err(
            "joint covariance requires an exact analytic Hessian; objective perturbation is forbidden"
                .to_string(),
        );
    };
    for (b, spec) in specs.iter().enumerate() {
        let (start, end) = ranges[b];
        let lambdas = per_block_log_lambdas[b].mapv(f64::exp);
        let mut s_lambda = Array2::<f64>::zeros((end - start, end - start));
        for (k, s) in spec.penalties.iter().enumerate() {
            s.add_scaled_to(lambdas[k], &mut s_lambda);
        }
        h.slice_mut(ndarray::s![start..end, start..end])
            .scaled_add(1.0, &s_lambda);
    }
    symmetrize_dense_in_place(&mut h);
    if use_exact_newton_strict_spd(family) {
        // #748: the strict posterior precision is `H + S_λ` AT THE CONVERGED
        // OPTIMUM. A δ-ridge inverse `(H + S_λ + δI)⁻¹` would mask a genuinely
        // non-PD curvature and report it as if it were the posterior
        // covariance, biasing every standard error. Instead: eigendecompose and
        // **reject** when the precision is genuinely indefinite (a real
        // fit-quality failure — the mode is not a strict maximum), and on the
        // PSD case return the honest positive-eigenspace pseudo-inverse (the
        // structural null space of a penalised model is a flat posterior
        // direction, not something to ridge away).
        let p = h.nrows();
        let (evals, _) = FaerEigh::eigh(&h, Side::Lower).map_err(|e| {
            format!("strict pseudo-laplace covariance eigendecomposition failed: {e}")
        })?;
        let max_abs_eval = evals.iter().fold(0.0_f64, |acc, &ev| acc.max(ev.abs()));
        let eps_np = f64::EPSILON * (p as f64) * (p as f64);
        let tol = (10.0 * eps_np * max_abs_eval).max(100.0 * f64::EPSILON);
        if let Some(&min_eval) = evals
            .iter()
            .filter(|&&ev| ev < -tol)
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        {
            let below = evals.iter().filter(|&&ev| ev < -tol).count();
            return Err(format!(
                "strict pseudo-laplace covariance: joint coefficient Hessian is non-PD at the \
                 converged optimum ({below} eigenvalue(s) below -tol, min(λ)={min_eval:.6e}, \
                 max|λ|={max_abs_eval:.6e}, tol={tol:.6e}); the mode is not a strict posterior \
                 maximum, so the reported covariance would be meaningless — fit-quality failure \
                 surfaced instead of δ-ridge masking (gam#748)"
            ));
        }
        pinv_positive_part(&h, effective_solverridge(options.ridge_floor))
    } else {
        match inverse_spdwith_retry(&h, effective_solverridge(options.ridge_floor), 8) {
            Ok(cov) => Ok(cov),
            Err(_) => pinv_positive_part(&h, effective_solverridge(options.ridge_floor)),
        }
    }
}


fn compute_joint_covariance_required<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    specs: &[ParameterBlockSpec],
    states: &[ParameterBlockState],
    per_block_log_lambdas: &[Array1<f64>],
    options: &BlockwiseFitOptions,
) -> Result<Option<Array2<f64>>, CustomFamilyError> {
    if !options.compute_covariance {
        return Ok(None);
    }
    compute_joint_covariance(family, specs, states, per_block_log_lambdas, options)
        .map(Some)
        .map_err(|e| CustomFamilyError::InvalidInput {
            context: "compute_joint_covariance_required",
            reason: format!("joint covariance computation failed: {e}"),
        })
}


/// Compute joint working-set geometry at convergence for ALO diagnostics.
fn compute_joint_geometry<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    specs: &[ParameterBlockSpec],
    states: &[ParameterBlockState],
    per_block_log_lambdas: &[Array1<f64>],
) -> Result<Option<FitGeometry>, String> {
    if specs.len() != per_block_log_lambdas.len() {
        return Ok(None);
    }
    if specs.len() == 1 {
        let eval = family.evaluate(states).ok();
        let Some(eval) = eval else {
            return Ok(None);
        };
        let spec = &specs[0];
        let lambdas = per_block_log_lambdas[0].mapv(f64::exp);
        // The penalized joint Hessian `H_pen = H_lik + Σ_k λ_k S_k` is the exact
        // mgcv quantity the trace edf `p − Σ_k λ_k·tr(H_pen⁻¹ S_k)` consumes. Two
        // single-block working-set shapes reach here:
        //
        // * `Diagonal` — IRLS/GLM families expose only the diagonal working
        //   weights, so the likelihood curvature is reconstructed as the
        //   Gauss–Newton gram `XᵀWX`.
        // * `ExactNewton` — coefficient-space exact-curvature families (CTN
        //   transformation-normal, …) already carry the dense negative
        //   log-likelihood Hessian `−∇²log L = H_lik` directly. Materialize it
        //   and add the penalties, so these families report inference / total
        //   edf instead of dropping geometry (and therefore inference) for the
        //   whole fit (#720).
        let (mut h, working_weights, working_response) = match eval.blockworking_sets.as_slice() {
            [
                BlockWorkingSet::Diagonal {
                    working_response,
                    working_weights,
                },
            ] => {
                let Some(h) = spec
                    .design
                    .xt_diag_x_signed_op(SignedWeightsView::from_array(working_weights))
                    .ok()
                else {
                    return Ok(None);
                };
                (h, working_weights.clone(), working_response.clone())
            }
            [BlockWorkingSet::ExactNewton { hessian, .. }] => {
                let h = hessian.to_dense();
                if h.nrows() != spec.design.ncols() || h.ncols() != spec.design.ncols() {
                    return Ok(None);
                }
                // The exact-Newton block carries no IRLS pseudo-data; the
                // trace edf reads only the penalized Hessian, and the
                // downstream IRLS covariance path is unused for these
                // families (they report dispersion = 1). Match the joint
                // multi-block branch's zero-length convention.
                let working_len = states.first().map(|state| state.eta.len()).unwrap_or(0);
                (h, Array1::zeros(working_len), Array1::zeros(working_len))
            }
            _ => return Ok(None),
        };
        for (k, s) in spec.penalties.iter().enumerate() {
            let s_dense = s.as_dense_cow();
            h.scaled_add(lambdas[k], &*s_dense);
        }
        // Exact-Newton families may return a Hessian assembled from directional
        // callbacks whose off-diagonal entries differ by floating-point order
        // or, for pseudo-Laplace tests, by a deliberately non-symmetric input
        // that is accepted only after symmetrization. Export the same symmetric
        // penalized Hessian used by the determinant/covariance path instead of
        // letting result assembly reject an otherwise valid fit geometry.
        symmetrize_dense_in_place(&mut h);
        return Ok(Some(FitGeometry {
            penalized_hessian: h.into(),
            working_weights,
            working_response,
        }));
    }

    let requires_explicit_joint_hessian = specs.iter().enumerate().any(|(idx, spec)| {
        custom_family_block_role(&spec.name, idx, specs.len())
            == crate::solver::estimate::BlockRole::LinkWiggle
    });
    let total_p: usize = specs.iter().map(|spec| spec.design.ncols()).sum();
    let Some(mut h) = exact_newton_joint_hessian_symmetrized(
        family,
        states,
        specs,
        total_p,
        "compute_joint_geometry",
    )?
    else {
        if requires_explicit_joint_hessian {
            return Err(
                "link-wiggle fits require an exact explicit joint Hessian for posterior sampling"
                    .to_string(),
            );
        }
        return Ok(None);
    };
    let ranges = block_param_ranges(specs);
    for (block_idx, spec) in specs.iter().enumerate() {
        let Some(block_log_lambdas) = per_block_log_lambdas.get(block_idx) else {
            return Ok(None);
        };
        let lambdas = block_log_lambdas.mapv(f64::exp);
        if lambdas.len() != spec.penalties.len() {
            return Ok(None);
        }
        let (start, end) = ranges[block_idx];
        let block_dim = end - start;
        for (penalty_idx, penalty) in spec.penalties.iter().enumerate() {
            let scale = lambdas[penalty_idx];
            if scale == 0.0 {
                continue;
            }
            let dense = penalty.as_dense_cow();
            if dense.nrows() == block_dim && dense.ncols() == block_dim {
                h.slice_mut(ndarray::s![start..end, start..end])
                    .scaled_add(scale, &*dense);
            } else if dense.nrows() == total_p && dense.ncols() == total_p {
                h.scaled_add(scale, &*dense);
            } else {
                return Ok(None);
            }
        }
    }
    let working_len = states.first().map(|state| state.eta.len()).unwrap_or(0);
    Ok(Some(FitGeometry {
        penalized_hessian: h.into(),
        working_weights: Array1::zeros(working_len),
        working_response: Array1::zeros(working_len),
    }))
}


pub fn fit_custom_family<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
) -> Result<crate::solver::estimate::UnifiedFitResult, CustomFamilyError> {
    fit_custom_family_with_rho_prior(family, specs, options, crate::types::RhoPrior::Flat)
}


/// Lift reduced-space `ParameterBlockState`s back to the raw block
/// dimensions described by `canonical.gauge`. Each block's
/// `beta` becomes `T_i · θ_i` (selection-T zeros dropped raw entries);
/// `eta = design · beta` is invariant under the transform, so the
/// reduced-space `eta` field carries through unchanged.
fn lift_block_states_to_raw(
    canonical: &crate::solver::identifiability_canonical::CanonicalSpecs,
    reduced: Vec<ParameterBlockState>,
) -> Vec<ParameterBlockState> {
    let theta_blocks: Vec<Array1<f64>> = reduced.iter().map(|s| s.beta.clone()).collect();
    let raw_betas = canonical.gauge.lift_block_betas(&theta_blocks);
    reduced
        .into_iter()
        .zip(raw_betas.into_iter())
        .map(|(state, beta_raw)| ParameterBlockState {
            beta: beta_raw,
            eta: state.eta,
        })
        .collect()
}


/// Lift a reduced-space conditional covariance / joint geometry pair
/// back to the raw coordinate system by sandwiching with the joint
/// block-diagonal transform `T_full = blockdiag(T_i)`. Selection-T
/// zero-pads the dropped raw rows/cols; the lifted Hessian is exactly
/// the post-canonicalisation Hessian as seen in raw coordinates and is
/// rank-deficient by construction along the dropped directions
/// (matching the inner-solve geometry the canonical step produced).
fn lift_fit_geometry_to_raw(
    canonical: &crate::solver::identifiability_canonical::CanonicalSpecs,
    covariance_conditional: Option<Array2<f64>>,
    geometry: Option<FitGeometry>,
) -> (Option<Array2<f64>>, Option<FitGeometry>) {
    let lifted_cov = covariance_conditional.map(|c| canonical.gauge.lift_covariance(&c));
    let lifted_geom = geometry.map(|g| {
        let h_red = g.penalized_hessian.into_array();
        let h_raw = canonical.gauge.lift_covariance(&h_red);
        FitGeometry {
            penalized_hessian: h_raw.into(),
            working_weights: g.working_weights,
            working_response: g.working_response,
        }
    });
    (lifted_cov, lifted_geom)
}


struct BlockwiseFitAssembly<'a> {
    rho_physical: Array1<f64>,
    covariance_conditional: Option<Array2<f64>>,
    geometry: Option<FitGeometry>,
    canonical: Option<&'a crate::solver::identifiability_canonical::CanonicalSpecs>,
    result_specs: &'a [ParameterBlockSpec],
    penalized_objective: f64,
    outer_iterations: usize,
    outer_gradient_norm: Option<f64>,
    criterion_certificate: Option<crate::solver::outer_strategy::CriterionCertificate>,
    outer_converged: bool,
    context: &'static str,
}


fn assemble_custom_family_fit_result(
    inner: BlockwiseInnerResult,
    assembly: BlockwiseFitAssembly<'_>,
) -> Result<crate::solver::estimate::UnifiedFitResult, CustomFamilyError> {
    let BlockwiseFitAssembly {
        rho_physical,
        covariance_conditional,
        geometry,
        canonical,
        result_specs,
        penalized_objective,
        outer_iterations,
        outer_gradient_norm,
        criterion_certificate,
        outer_converged,
        context,
    } = assembly;
    let lambdas = rho_physical.mapv(f64::exp);
    let log_lambdas = lambdas.mapv(|v| v.max(1e-300).ln());
    let (block_states, covariance_conditional, geometry, precomputed_edf) =
        if let Some(canonical) = canonical {
            let precomputed_edf = reduced_blockwise_edf(geometry.as_ref(), canonical, &lambdas);
            let block_states = lift_block_states_to_raw(canonical, inner.block_states);
            let (covariance_conditional, geometry) =
                lift_fit_geometry_to_raw(canonical, covariance_conditional, geometry);
            (
                block_states,
                covariance_conditional,
                geometry,
                precomputed_edf,
            )
        } else {
            (inner.block_states, covariance_conditional, geometry, None)
        };

    blockwise_fit_from_parts(
        BlockwiseFitResultParts {
            block_states,
            log_likelihood: inner.log_likelihood,
            log_lambdas,
            lambdas,
            covariance_conditional,
            stable_penalty_term: 2.0 * inner.penalty_value,
            penalized_objective,
            outer_iterations,
            outer_gradient_norm,
            criterion_certificate,
            inner_cycles: inner.cycles,
            outer_converged,
            geometry,
            precomputed_edf,
        },
        result_specs,
    )
    .map_err(|reason| CustomFamilyError::Optimization { context, reason })
}


/// Install the channel-aware `AdditiveBlockJacobian` callbacks declared by a
/// family's [`CustomFamily::output_channel_assignment`].
///
/// Multi-output families that build their specs by hand (or through the
/// low-level `fit_custom_family` API) declare their per-block output channel
/// here so the pre-fit identifiability audit routes channel-aware instead of
/// mistaking a shared covariate basis for cross-block aliases (#558). Blocks
/// that already carry an explicit `jacobian_callback` are left untouched
/// (the family wired its own, possibly β-dependent, multi-output Jacobian).
///
/// Returns `None` when the family declares no assignment (single-output flat
/// route, the default) so the caller can keep borrowing the original specs
/// without an allocation.
fn wire_output_channels<F: CustomFamily + ?Sized>(
    family: &F,
    specs: &[ParameterBlockSpec],
) -> Result<Option<Vec<ParameterBlockSpec>>, CustomFamilyError> {
    validate_blockspecs(specs)?;
    let Some(channels) = family.output_channel_assignment(specs) else {
        return Ok(None);
    };
    if channels.len() != specs.len() {
        return Err(CustomFamilyError::DimensionMismatch {
            reason: format!(
                "output_channel_assignment returned {} channels for {} blocks",
                channels.len(),
                specs.len(),
            ),
        });
    }
    let n_family_outputs = channels.iter().copied().max().map(|m| m + 1).unwrap_or(1);
    if n_family_outputs <= 1 {
        // A single output channel is exactly the flat route — nothing to wire.
        return Ok(None);
    }
    // When every block already carries an explicit (family-wired) callback,
    // the channel-aware route is already taken — avoid cloning the specs.
    if specs.iter().all(|s| s.jacobian_callback.is_some()) {
        return Ok(None);
    }
    let mut wired = specs.to_vec();
    for (idx, spec) in wired.iter_mut().enumerate() {
        // Respect a family-supplied callback (e.g. multinomial / location-scale
        // already wire their own multi-output, possibly β-dependent Jacobian).
        if spec.jacobian_callback.is_some() {
            continue;
        }
        let own_output = channels[idx];
        // The block's effective design at β=0 (with no callback) is exactly
        // its linear design — the additive-block Jacobian for an `η_r = X_r β_r`
        // channel.
        let dense = spec.effective_design("wire_output_channels").map_err(|e| {
            CustomFamilyError::DimensionMismatch {
                reason: format!("block {idx} effective design for channel wiring: {e}"),
            }
        })?;
        spec.jacobian_callback = Some(Arc::new(AdditiveBlockJacobian {
            design: dense,
            own_output,
            n_family_outputs,
        }));
    }
    Ok(Some(wired))
}


/// True iff an outer-smoothing `Err` is a POST-AUDIT NUMERICAL pathology that
/// the never-fail posterior-sampling rung can recover from (gam#860), rather
/// than an ill-posed input that must keep raising.
///
/// All structural guards (the #531-class identifiability audit, the #789B
/// zero-events guard, the #859 cross-fit alignment check) raise BEFORE the outer
/// solver runs, so by the time the outer optimizer reports "no candidate seeds
/// passed outer startup validation" (every seed rejected during exact-eval
/// validation, e.g. the #787 kappa-driven penalty-topology dim-mismatch that
/// surfaces as a non-finite cost) the design is structurally well-posed and a
/// posterior mode exists to sample about. Those two signatures are the
/// escalatable ones. Any other `Err` (a genuine solver contract violation,
/// dimension error, etc.) keeps the hard raise.
fn outer_startup_failure_is_escalatable(err: &EstimationError) -> bool {
    match err {
        EstimationError::RemlOptimizationFailed(message) => {
            message.contains("no candidate seeds passed outer startup validation")
                || message.contains("objective returned a non-finite cost")
                // Data-driven inner non-convergence on a structurally-audited design:
                // the coupled exact-joint Newton path could not drive a weakly-identified
                // block's penalized stationarity residual below tol at every screened seed
                // (the #787 weak marginal/logslope-coupling KKT-flooring regime). This
                // surfaces as a hard `Err` from the inner solve (rather than the
                // `Ok(!inner_converged)` retreat sentinel), so when it rejects every seed
                // BEFORE the outer optimizer starts it would otherwise dead-end short of
                // the post-run escalation rung. It is a post-audit NUMERICAL pathology, not
                // an ill-posed input — the best inner mode reached during screening is a
                // usable posterior mode — so route it into the same never-fail escalation
                // (gam#860).
                //
                // Both coupled-exact-joint non-convergence signatures qualify: the
                // pre-budget "exited the joint Newton path before convergence" exit and
                // the "exhausted the joint Newton budget without KKT convergence" exit are
                // the same #787-class weak-identification floor reached two ways.
                //
                // The SAME prefixes are also emitted for GENUINELY STRUCTURAL cert
                // refusals (the diagnosis is carried in the trailing `; diagnosis: <label>`
                // slot of the bubbled error). Those — a rank-deficient joint design, an
                // unresolved active set, or a cross-block alias surfaced at fit time — are
                // NOT recoverable by sampling about the mode (the mode itself is
                // degenerate), so they must keep hard-raising. We therefore escalate the
                // coupled-joint failure only when it carries no structural diagnosis label.
                || ((message
                    .contains("coupled exact-joint inner solve exited the joint Newton path")
                    || message.contains(
                        "coupled exact-joint inner solve exhausted the joint Newton budget",
                    ))
                    && !message.contains("diagnosis: rank_deficient_H_pen")
                    && !message.contains("diagnosis: active_set_incomplete")
                    && !message.contains("diagnosis: aliasing_detected_at_fit"))
        }
        _ => false,
    }
}


/// Minimum effective degrees of freedom a penalized term must retain in the
/// outer λ-selection. One effective dimension is the smallest non-arbitrary
/// floor: it asserts the penalized component must explain at least ONE effective
/// direction of its own range space, i.e. it has not collapsed entirely onto its
/// unpenalized polynomial null space. It is NOT a tuning constant — `1.0` is the
/// boundary between "the smooth contributes" and "the smooth is statistically
/// indistinguishable from its null-space limit".
const EFFECTIVE_DF_FLOOR: f64 = 1.0;


/// Unit-weight effective degrees of freedom of a single penalized term as a
/// function of `ρ = log λ`, expressed through the design/penalty generalized
/// eigenvalues `γ_j` on the penalty range space:
///
/// ```text
/// edf(ρ) = Σ_j γ_j / (γ_j + e^ρ),   γ_j = (design range curvature)_j / (penalty)_j.
/// ```
///
/// This is the data-FREE structural edf: it uses the design column Gram `XᵀX`
/// (unit weights), NOT the family's Fisher weight, so it is the same regardless
/// of where the inner solve sits on a near-flat Fisher surface. It is the
/// quantity whose collapse the #715/#684 over-shrinkage describes — when the
/// Fisher curvature vanishes the REML objective flattens in ρ and the optimizer
/// lets λ drift past the point where this structural edf falls below the floor.
fn unit_weight_term_edf(gammas: &[f64], rho: f64) -> f64 {
    let lambda = rho.exp();
    gammas
        .iter()
        .map(|&g| if g > 0.0 { g / (g + lambda) } else { 0.0 })
        .sum()
}


/// Generalized eigenvalues `γ_j` of the design column Gram `G = XᵀX` against the
/// penalty `S` on `range(S)`, computed structurally (unit weights).
///
/// These are the eigenvalues of the pencil `(UᵀG U, D)` where `S = U D Uᵀ` and
/// the index runs over `range(S)` (the positive eigenvalues `d_j` of `S`).
/// Equivalently they are the eigenvalues of the symmetric matrix
///
/// ```text
/// B = D^{-1/2} (Uᵀ G U) D^{-1/2}   restricted to range(S),
/// ```
///
/// with `D = diag(d_j)` over the range and `U` the corresponding penalty
/// eigenvectors. With these `γ_j` the structural effective df is the EXACT
/// trace identity
///
/// ```text
/// Σ_j γ_j/(γ_j + λ) = tr{ G (G + λ S)⁻¹ }   for all λ > 0.
/// ```
///
/// This is NOT a per-direction Rayleigh quotient `(u_jᵀ G u_j)/d_j`: that would
/// keep only the diagonal of `B` and is correct only when `G` and `S` commute
/// (are simultaneously diagonalizable). Smooth Gram/penalty pairs generally do
/// not commute, so the off-diagonal coupling of `B` must be retained — it is
/// what makes the eigenvalue sum match the trace identity above.
///
/// Returns `None` (caller falls back to the uniform ρ bound) whenever the
/// geometry cannot be materialized safely as a `p×p` block-local pair — Kronecker
/// penalties are expanded, but `Blockwise`/total-dim penalties whose dense form
/// is not `p×p` are skipped rather than risk a mis-projected curvature that could
/// bias the REML selection.
fn design_penalty_range_gammas(design: &DesignMatrix, penalty: &PenaltyMatrix) -> Option<Vec<f64>> {
    let p = design.ncols();
    if p == 0 {
        return None;
    }
    let s_dense = penalty.to_dense();
    if s_dense.nrows() != p || s_dense.ncols() != p {
        // Blockwise/total-dim layout or shape mismatch: not safely projectable
        // here. Fall back to the uniform bound.
        return None;
    }
    let x = design.to_dense();
    if x.ncols() != p {
        return None;
    }
    let gram = x.t().dot(&x);
    // Eigendecompose the penalty to find its range space S = U D Uᵀ.
    let (s_evals, s_evecs) = s_dense.eigh(Side::Lower).ok()?;
    let s_max = s_evals.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
    if !(s_max > 0.0) {
        return None;
    }
    let s_thresh = positive_eigenvalue_threshold(s_evals.as_slice()?);
    // Collect the range-space columns U_r (penalty eigenvectors with d_j above
    // the numerical-zero threshold) and their inverse square-root weights
    // d_j^{-1/2}. Directions in ker(S) are dropped: they are unpenalized and do
    // not enter the structural edf of this term.
    let mut range_cols: Vec<usize> = Vec::new();
    let mut inv_sqrt_d: Vec<f64> = Vec::new();
    for (j, &dj) in s_evals.iter().enumerate() {
        if dj <= s_thresh {
            continue; // null space of S: not a penalized direction.
        }
        range_cols.push(j);
        inv_sqrt_d.push(1.0 / dj.sqrt());
    }
    let r = range_cols.len();
    if r == 0 {
        return None;
    }
    // Form U_r (p×r) and the symmetric pencil matrix
    //   B = D_r^{-1/2} (U_rᵀ G U_r) D_r^{-1/2}   (r×r),
    // whose eigenvalues are the generalized eigenvalues of (UᵀGU, D) on
    // range(S). Scaling U_r's columns by d_j^{-1/2} up front gives
    //   Y = U_r D_r^{-1/2}  (p×r),   B = Yᵀ G Y,
    // which is symmetric by construction (Gram of G in the Y-columns).
    let mut y = Array2::<f64>::zeros((p, r));
    for (col, (&src, &w)) in range_cols.iter().zip(inv_sqrt_d.iter()).enumerate() {
        let u = s_evecs.column(src);
        for row in 0..p {
            y[(row, col)] = u[row] * w;
        }
    }
    let b = y.t().dot(&gram).dot(&y);
    // Symmetrize defensively against round-off before the symmetric solver, then
    // take eigenvalues. These are the γ_j (data-free, unit-weight).
    let mut b_sym = b.clone();
    for i in 0..r {
        for j in (i + 1)..r {
            let avg = 0.5 * (b_sym[(i, j)] + b_sym[(j, i)]);
            b_sym[(i, j)] = avg;
            b_sym[(j, i)] = avg;
        }
    }
    let (b_evals, _) = b_sym.eigh(Side::Lower).ok()?;
    let mut gammas = Vec::with_capacity(r);
    for &gj in b_evals.iter() {
        // A penalized direction with no design support has γ→0: edf→0 for any
        // λ>0, so it cannot be floored by bounding ρ. Clamp tiny negative
        // round-off to 0; it never contributes to the retained df sum.
        if gj.is_finite() && gj > 0.0 {
            gammas.push(gj);
        } else {
            gammas.push(0.0);
        }
    }
    if gammas.is_empty() {
        return None;
    }
    Some(gammas)
}


/// Per-outer-coordinate ρ UPPER bound enforcing the effective-df floor.
///
/// For each penalized term, the structural unit-weight edf `Σ_j γ_j/(γ_j+e^ρ)`
/// is monotone decreasing in ρ. The bound is the ρ at which it equals
/// `EFFECTIVE_DF_FLOOR` (when the term's max attainable edf exceeds the floor),
/// found by bisection on the closed-form edf. Tied coordinates (shared precision
/// label) take the TIGHTEST (smallest) per-term bound: the shared λ must retain
/// the floor for EVERY contributing term, so the binding constraint is the most
/// restrictive one — relaxing to a looser term's bound would let some other term
/// fall below its floor. Every coordinate is additionally capped at the caller's
/// uniform `ceiling` so this can only TIGHTEN, never loosen, the existing bound.
///
/// This enters ONLY the λ-selection domain. The inner β solve is exact
/// CONDITIONAL on the selected λ, so there is no per-λ approximation (same
/// discipline as the #747 solver-only ridge). It is NOT, however, a bias-free
/// no-op: whenever the unconstrained REML optimum lies beyond this upper bound,
/// the bound changes the SELECTED λ, and the selected λ changes the fitted
/// β̂ = argmin{−ℓ + ½λ βᵀSβ} (∂β̂/∂λ = −(H + λS)⁻¹ S β̂ ≠ 0). The floor is an
/// explicit smoothing-regularization constraint on the λ-selection — it
/// deliberately moves the estimate away from the (flat-Fisher) null-space
/// collapse, not a transparent reparameterization. It is the λ-upper-side dual
/// of the #752
/// full-subspace logdet work — there the value/gradient subspace was fixed on the
/// λ→∞ side of a near-collinear block; here the selection domain is bounded so a
/// flat Fisher surface cannot push a term past null-space collapse (#715/#684).
fn effective_df_floor_rho_upper_bounds(
    specs: &[ParameterBlockSpec],
    layout: &PenaltyLabelLayout,
    n_rho: usize,
    ceiling: f64,
) -> Array1<f64> {
    let mut upper = Array1::<f64>::from_elem(n_rho, ceiling);
    let mut physical = 0usize;
    for spec in specs {
        for penalty in &spec.penalties {
            let outer = layout.physical_to_outer.get(physical).copied().flatten();
            physical += 1;
            let Some(outer) = outer else {
                continue; // fixed penalty: not an outer coordinate.
            };
            let Some(gammas) = design_penalty_range_gammas(&spec.design, penalty) else {
                continue; // un-projectable geometry: keep the uniform ceiling.
            };
            // Maximum attainable structural edf (ρ → −∞) is the number of
            // design-supported penalized directions. If it cannot reach the
            // floor even unpenalized, the floor is not enforceable for this term
            // (a single-dimension range space with the floor at its own cap), so
            // keep the uniform ceiling.
            let edf_max = unit_weight_term_edf(&gammas, f64::NEG_INFINITY);
            if !(edf_max > EFFECTIVE_DF_FLOOR) {
                continue;
            }
            // Bisect for ρ* with edf(ρ*) = floor on [−ceiling, ceiling]; edf is
            // monotone decreasing in ρ. If edf at the ceiling still exceeds the
            // floor, the uniform ceiling already retains enough df — keep it.
            if unit_weight_term_edf(&gammas, ceiling) >= EFFECTIVE_DF_FLOOR {
                continue;
            }
            let mut lo = -ceiling;
            let mut hi = ceiling;
            for _ in 0..64 {
                let mid = 0.5 * (lo + hi);
                if unit_weight_term_edf(&gammas, mid) >= EFFECTIVE_DF_FLOOR {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            let rho_star = 0.5 * (lo + hi);
            // Tied coordinates: take the tightest (smallest) bound across terms,
            // so every term sharing this λ retains at least the floor.
            let slot = &mut upper[outer];
            if rho_star > -ceiling && rho_star < *slot {
                *slot = rho_star;
            }
        }
    }
    upper
}


pub fn fit_custom_family_with_rho_prior<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
    rho_prior: crate::types::RhoPrior,
) -> Result<crate::solver::estimate::UnifiedFitResult, CustomFamilyError> {
    // Multi-output families that omitted the per-block channel callback get it
    // installed here from their declared `output_channel_assignment`, so the
    // identifiability audit routes channel-aware (single source of truth for
    // the channel-wiring; no per-test/per-builder duplication — #558).
    let wired = wire_output_channels(family, specs)?;
    let raw_specs: &[ParameterBlockSpec] = wired.as_deref().unwrap_or(specs);
    validate_blockspecs(raw_specs)?;

    // Pre-fit cross-block identifiability canonicalisation. Every
    // blockwise fit path in the tree (standard, gaussian/binomial
    // location-scale, survival, BMS, transformation-normal, custom
    // families) reaches this entry point with a finalised
    // `ParameterBlockSpec` list, so wiring the canonicalisation here
    // covers all four `solver::workflow.rs` entry points plus every
    // direct caller of `fit_custom_family` without each family needing
    // its own canonicalisation hook.
    //
    // Contract: specs arrive *after* `nullspace-lead`'s
    // `joint_null_rotation` absorption. The canonical step inspects
    // post-rotation columns only, runs the joint RRQR identifiability
    // audit, and converts attributed cross-block drops into a per-block
    // selection transform `T_i`. The inner solve runs in the reduced
    // coordinate space; coefficients and joint geometry are lifted back
    // to the raw space at result assembly via `T_i` and the joint
    // block-diagonal `T_full = blockdiag(T_i)`.
    //
    // An audit that is fatal *without* attributed drops (the >2-way
    // structural alias case where RRQR couldn't pin redundancy onto a
    // single block/column) still aborts: silently absorbing it would
    // change model semantics beyond what canonicalisation can repair.
    // Per the panic-vs-Err contract: never panic mid-construction.
    let canonical_started = std::time::Instant::now();
    let canonical_n_rows = raw_specs.first().map(|s| s.design.nrows()).unwrap_or(0);
    let canonical_n_cols_raw: usize = raw_specs.iter().map(|s| s.design.ncols()).sum();
    log::info!(
        "[STAGE] identifiability canonicalise: start blocks={} n={} p_total_raw={}",
        raw_specs.len(),
        canonical_n_rows,
        canonical_n_cols_raw,
    );
    let canonical =
        crate::solver::identifiability_canonical::canonicalize_for_identifiability(raw_specs)?;
    let canonical_n_cols_red: usize = canonical
        .reduced_specs
        .iter()
        .map(|s| s.design.ncols())
        .sum();
    log::info!(
        "[STAGE] identifiability canonicalise: end elapsed={:.3}s alias_pairs={} dropped_cols={} \
         p_total_raw={} p_total_reduced={} fatal_attributed={}",
        canonical_started.elapsed().as_secs_f64(),
        canonical.audit.aliased_pairs.len(),
        canonical.audit.dropped_columns.len(),
        canonical_n_cols_raw,
        canonical_n_cols_red,
        canonical.audit.fatal,
    );
    if !canonical.audit.aliased_pairs.is_empty() {
        log::info!("[identifiability audit] {}", canonical.audit.summary);
        // Aggregate by (block_a, block_b) so the log stays bounded by the
        // block-pair count rather than the quadratic direction-pair count
        // — a few wide blocks alone produce 100+ pair-lines and bury the
        // useful structural signal. INFO carries the cluster shape (count,
        // overlap range, perfect-collinearity count); DEBUG prints the
        // worst three sample pairs per cluster for forensic users.
        let mut by_pair: BTreeMap<(&str, &str), Vec<&_>> = BTreeMap::new();
        for pair in &canonical.audit.aliased_pairs {
            by_pair
                .entry((pair.block_a.as_str(), pair.block_b.as_str()))
                .or_default()
                .push(pair);
        }
        for ((a, b), pairs) in &by_pair {
            let count = pairs.len();
            let max = pairs
                .iter()
                .map(|p| p.overlap)
                .fold(f64::NEG_INFINITY, f64::max);
            let min = pairs
                .iter()
                .map(|p| p.overlap)
                .fold(f64::INFINITY, f64::min);
            let near_one = pairs.iter().filter(|p| p.overlap >= 0.9999).count();
            log::info!(
                "[identifiability audit] alias-cluster {a} ~ {b}: {count} direction-pair{plural} \
                 (overlap {min:.4}..{max:.4}; {near_one} ≥0.9999)",
                plural = if count == 1 { "" } else { "s" },
            );
        }
        if log::log_enabled!(log::Level::Debug) {
            for ((a, b), pairs) in &by_pair {
                let mut sorted = pairs.clone();
                sorted.sort_by(|p, q| {
                    q.overlap
                        .partial_cmp(&p.overlap)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                for pair in sorted.iter().take(3) {
                    log::debug!(
                        "[identifiability audit]   sample {a}[{ai}] ~ {b}[{bi}] overlap={ov:.4}",
                        ai = pair.direction_a,
                        bi = pair.direction_b,
                        ov = pair.overlap,
                    );
                }
            }
        }
    }
    for drop in &canonical.audit.dropped_columns {
        log::info!(
            "[identifiability audit] dropped: block='{}' local_col={} ({})",
            drop.block,
            drop.column,
            drop.reason,
        );
    }
    let specs: &[ParameterBlockSpec] = &canonical.reduced_specs;
    let penalty_counts = validate_blockspecs(specs)?;

    let label_layout = penalty_label_layout(specs, penalty_counts.clone())?;
    let rho0 = label_layout.initial_rho.clone();
    let (persistent_warm_start_key, persistent_warm_start) =
        load_persistent_custom_family_warm_start::<F>(family, specs, options, rho0.len());

    if rho0.is_empty() {
        let physical_rho0 = expand_labeled_log_lambdas(&rho0, &label_layout)?;
        let per_block = split_labeled_log_lambdas(&rho0, &label_layout)?;
        let mut inner = inner_blockwise_fit(
            family,
            specs,
            &per_block,
            options,
            persistent_warm_start.as_ref(),
        )?;
        refresh_all_block_etas(family, specs, &mut inner.block_states)?;
        let covariance_conditional = compute_joint_covariance_required(
            family,
            specs,
            &inner.block_states,
            &per_block,
            options,
        )?;
        let reml_term = if options.use_remlobjective {
            0.5 * (inner.block_logdet_h - inner.block_logdet_s)
        } else {
            0.0
        };
        let geometry = compute_joint_geometry(family, specs, &inner.block_states, &per_block)
            .map_err(|reason| CustomFamilyError::Optimization {
                context: "fit_custom_family no-smoothing joint geometry",
                reason,
            })?;
        let penalized_objective = checked_penalizedobjective(
            inner.log_likelihood,
            inner.penalty_value,
            reml_term,
            "custom-family fit without smoothing parameters",
        )
        .map_err(|reason| CustomFamilyError::Optimization {
            context: "fit_custom_family no-smoothing penalized objective",
            reason,
        })?;
        let warm_start = constrained_warm_start_from_inner(&rho0, &inner);
        store_persistent_custom_family_warm_start(
            persistent_warm_start_key.as_deref(),
            specs,
            &warm_start,
        );
        let inner_converged = inner.converged;
        return assemble_custom_family_fit_result(
            inner,
            BlockwiseFitAssembly {
                rho_physical: physical_rho0,
                covariance_conditional,
                geometry,
                canonical: Some(&canonical),
                result_specs: raw_specs,
                penalized_objective,
                outer_iterations: 0,
                outer_gradient_norm: None,
                criterion_certificate: None,
                outer_converged: inner_converged,
                context: "fit_custom_family no-smoothing result assembly",
            },
        );
    }

    // Exact Hessians are primary whenever the assembled family can supply them.
    // If a particular outer step is ill-conditioned, strategy fallback handles
    // the downgrade; we do not suppress second-order capability preemptively
    // based on the presence of a wiggle block.
    if options.inner_max_cycles <= 1 && options.outer_max_iter <= 1 {
        log::info!(
            "[OUTER] custom family: skipping smoothing outer solve for explicit one-cycle inner probe"
        );
        let per_block = split_labeled_log_lambdas(&rho0, &label_layout)?;
        let mut inner = inner_blockwise_fit(family, specs, &per_block, options, None)?;
        refresh_all_block_etas(family, specs, &mut inner.block_states).map_err(|reason| {
            CustomFamilyError::Optimization {
                context: "fit_custom_family one-cycle eta refresh",
                reason,
            }
        })?;
        let penalized_objective = inner_penalized_objective(
            &inner,
            include_exact_newton_logdet_h(family, options),
            include_exact_newton_logdet_s(family, options),
            "custom-family explicit one-cycle inner probe",
        )
        .map_err(|reason| CustomFamilyError::Optimization {
            context: "fit_custom_family one-cycle penalized objective",
            reason,
        })?;
        let physical_rho0 = expand_labeled_log_lambdas(&rho0, &label_layout)?;
        let inner_converged = inner.converged;
        return assemble_custom_family_fit_result(
            inner,
            BlockwiseFitAssembly {
                rho_physical: physical_rho0,
                covariance_conditional: None,
                geometry: None,
                canonical: Some(&canonical),
                result_specs: raw_specs,
                penalized_objective,
                outer_iterations: 0,
                outer_gradient_norm: Some(0.0),
                criterion_certificate: None,
                outer_converged: inner_converged,
                context: "fit_custom_family one-cycle result assembly",
            },
        );
    }

    use crate::estimate::EstimationError;
    use crate::solver::outer_strategy::{FallbackPolicy, OuterEval, OuterEvalOrder, OuterProblem};

    let screening_cap = Arc::new(AtomicUsize::new(0));
    let outer_inner_cap = options
        .outer_inner_max_iterations
        .clone()
        .unwrap_or_else(|| Arc::new(AtomicUsize::new(options.inner_max_cycles.max(1))));
    outer_inner_cap.store(options.inner_max_cycles.max(1), Ordering::Relaxed);
    let mut outer_options = options.clone();
    outer_options.screening_max_inner_iterations = Some(Arc::clone(&screening_cap));
    outer_options.outer_inner_max_iterations = Some(Arc::clone(&outer_inner_cap));

    let n_rho = rho0.len();
    let (cap_gradient, cap_hessian) =
        custom_family_outer_derivatives(family, specs, &outer_options);
    let derivative_policy = family.outer_derivative_policy(specs, 0, &outer_options);
    let hessian = cap_hessian;
    let need_outer_hessian = hessian.is_analytic();
    log::info!(
        "[OUTER] custom family derivative-policy: n_params={} gradient={:?} hessian={:?} capability={:?} requested_outer_hessian={} predicted_gradient_work={} predicted_hessian_work={} inner_hvp_available={} outer_hvp_available={} outer_dense_available={}",
        n_rho,
        cap_gradient,
        hessian,
        derivative_policy.capability,
        need_outer_hessian,
        derivative_policy.predicted_gradient_work,
        derivative_policy.predicted_hessian_work,
        family.inner_coefficient_hessian_hvp_available(specs),
        family.outer_hyper_hessian_hvp_available(specs),
        family.outer_hyper_hessian_dense_available(specs),
    );
    let outer_max_iter = cost_gated_first_order_max_iter(
        options.outer_max_iter,
        family.coefficient_gradient_cost(specs),
        need_outer_hessian,
    );
    let bfgs_step_cap = first_order_bfgs_loglambda_step_cap(need_outer_hessian);
    if outer_max_iter < options.outer_max_iter {
        log::info!(
            "[OUTER] custom family: first-order work gate reduced outer_max_iter {} -> {}",
            options.outer_max_iter,
            outer_max_iter,
        );
    }
    // EFS / HybridEfs structural property (`H^{-1/2} B_k H^{-1/2} ≽ 0` plus a
    // parameter-independent nullspace, Wood-Fasiolo) fails for multi-block
    // families whose joint likelihood Hessian depends on β. Disable
    // fixed-point only for genuinely first-order capabilities; exact-Hessian
    // capabilities route to ARC before EFS is considered.
    let multi_block_beta_dependent =
        specs.len() > 1 && family.exact_newton_joint_hessian_beta_dependent();
    // Exact-Hessian plans must fail on their own terms rather than silently
    // retrying on a quasi-Newton surface. First-order-only families keep the
    // automatic cascade because there is no second-order geometry to discard.
    let fallback_policy = if need_outer_hessian {
        FallbackPolicy::Disabled
    } else {
        FallbackPolicy::Automatic
    };
    // Calibrate the outer solver to the n-scaled profiled REML/LAML objective.
    // The profiled criterion is a sum over n observations, so |f| ~ O(n) for
    // every family. Without this calibration the outer ARC/BFGS:
    //   (a) uses a bare absolute gradient floor of `outer_tol ≈ 1e-5` — this
    //       IS achievable at scale but forces the optimizer to iterate until
    //       |g| ≤ 1e-5 even when |f| ~ 200 and τ·(1+|f|) ~ 2e-3 already
    //       signals convergence in the relative-to-cost sense; and
    //   (b) ARC's initial trust-region regularization `σ₀=1` and default
    //       operator trust radius `τ₀=1` reference the wrong curvature
    //       magnitude — the first ARC step overshoots when the Hessian is
    //       O(n) and the trust radius is O(1).
    // Mirroring the spatial exact-joint outer fix (#1053/#1066/#1069) and
    // the primary REML outer (solver/estimate.rs) for the custom-family path.
    let n_obs = specs.first().map(|s| s.design.nrows()).unwrap_or(0);
    let p_total: usize = specs.iter().map(|s| s.design.ncols()).sum();
    let problem = OuterProblem::new(n_rho)
        .with_gradient(cap_gradient)
        .with_hessian(hessian)
        .with_disable_fixed_point(multi_block_beta_dependent)
        .with_fallback_policy(fallback_policy)
        .with_tolerance(options.outer_tol)
        .with_max_iter(outer_max_iter)
        .with_bfgs_step_cap(bfgs_step_cap)
        .with_seed_config(family.outer_seed_config(n_rho))
        .with_initial_rho(rho0.clone())
        .with_screen_initial_rho(options.screen_initial_rho)
        // n-scaled profiled-criterion calibration: absolute gradient floor =
        // max(outer_tol, n·1e-9), ARC σ₀ = 0.25, operator trust radius = 4.0.
        // Mirrors the primary REML outer (solver/estimate.rs) and the spatial
        // exact-joint path.
        .with_objective_scale(if n_obs > 0 { Some(n_obs as f64) } else { None })
        .with_problem_size(n_obs, p_total.max(1))
        .with_arc_initial_regularization(if n_obs > 0 { Some(0.25) } else { None })
        .with_operator_initial_trust_radius(if n_obs > 0 { Some(4.0) } else { None })
        // Per-coordinate ρ box bounds. The uniform ceiling of 10 is the
        // belt-and-suspenders cap: λ = exp(10) ≈ 22k is already extremely strong
        // shrinkage, and the bound keeps the optimizer out of the dead-flat
        // λ ≈ 10⁹ region where ARC's quadratic model breaks down, the retry-stall
        // detector fires, and downstream empty-block_states crashes surface.
        //
        // ON TOP of that uniform ceiling, each penalized term's UPPER bound is
        // tightened to the ρ at which its structural (unit-weight) effective df
        // would fall to one — the EFFECTIVE_DF_FLOOR. Near a flat Fisher surface
        // (multinomial simplex boundary diag(p)−ppᵀ→0, #715; Gaussian log-σ on a
        // gently-varying scale, #684) the REML criterion loses ρ-curvature and
        // the optimizer would otherwise let some λ_{class,term} drift past the
        // point where the term collapses onto its unpenalized polynomial null
        // space, over-smoothing the cubic/sigmoid/log-σ signal below the mature
        // reference. The floor is derived from the penalty RANGE-SPACE
        // eigenstructure (design/penalty generalized eigenvalues), not from the
        // vanishing Fisher weight, and enters ONLY the λ-selection domain — the
        // inner β solve at the selected ρ is unchanged and exact, so the
        // converged β is unbiased (cf. the #747 solver-only ridge). This is the
        // λ-upper-side dual of the #752 full-subspace logdet work.
        .with_bounds(
            Array1::<f64>::from_elem(n_rho, -10.0),
            effective_df_floor_rho_upper_bounds(specs, &label_layout, n_rho, 10.0),
        );
    // Install the seed-screening cap only when initial-rho screening is
    // wanted. A caller that pins an already-identified `initial_rho` and
    // opts out (`screen_initial_rho == false`) leaves the OuterConfig
    // screening cap `None`, so `should_screen_seeds` short-circuits and the
    // screening cascade never runs. This is the lever the survival
    // constant-scale (parametric-AFT) regime uses: its time-warp ρ seed is
    // pinned AT the inner ρ box bound (the affine-baseline limit) on a
    // dead-flat, statistically-unidentified time ridge where every capped
    // proxy fit collapses to non-finite cost and the cascade escalates to a
    // full uncapped inner solve per seed on the near-singular Hessian — the
    // multi-minute no-iteration-log stall (#736, #735, #721). With the cap
    // unset, the pinned seed flows straight to the outer solver, which
    // certifies box-constraint stationarity at iteration 0. Every other
    // custom-family caller defaults `screen_initial_rho = true` and keeps
    // full screening; genuinely flexible scale/spatial survival fits carry
    // log-sigma penalties, never set the flag false, and screen normally.
    let problem = if options.screen_initial_rho {
        problem.with_screening_cap(Arc::clone(&screening_cap))
    } else {
        problem
    };
    // Attach the workflow-level warm-start session if one was threaded
    // through. This makes the custom-family outer optimizer (BFGS / ARC
    // depending on derivative capabilities) use the same persistent
    // cache infrastructure as standard REML — every accepted outer step
    // is checkpointed to disk, every fit starts by consulting the disk
    // for a prior best iterate. Without this, every survival-marginal-
    // slope / GAMLSS / latent fit starts cold even when a converged ρ
    // from a near-identical prior fit is sitting in `~/.cache/gam/warm`.
    let problem = if let Some(session) = options.cache_session.clone() {
        let key_hex = session.key().to_hex();
        log::info!(
            "[CACHE] attach key={}.. family-tag={} backend=outer-strategy mirrors={}",
            &key_hex[..8.min(key_hex.len())],
            std::any::type_name::<F>()
                .rsplit("::")
                .next()
                .unwrap_or("?"),
            options.cache_mirror_sessions.len(),
        );
        let mut p = problem.with_cache_session(session);
        if !options.cache_mirror_sessions.is_empty() {
            p = p.with_cache_mirror_sessions(options.cache_mirror_sessions.clone());
        }
        p
    } else {
        problem
    };

    // Robustness is unconditional, so escalation is always armed: the inner-non-
    // convergence branch inside `eval_outer` marks a trial rho *infeasible*
    // (recoverable) rather than hard-erroring, letting the outer optimizer retreat
    // and the run reach the terminal HMC sampling rung instead of dead-ending
    // before it (the gap `verify` located at this site).
    let eval_outer = |outer: &mut CustomOuterState,
                      rho: &Array1<f64>,
                      order: OuterEvalOrder|
     -> Result<OuterEval, EstimationError> {
        let warm_ref = screened_outer_warm_start(outer.warm_cache.as_ref(), rho);
        // Genuinely value-only fulfilment (#979). A `Value` request — issued only
        // by the continuation pre-warm and outer cost probes — never consumes the
        // outer gradient. Routing it through the value+gradient assembly below
        // paid a full coupled-joint LAML gradient (the k²·n·p² marginal/log-slope
        // outer-derivative) at EVERY continuation step purely to carry the warm β
        // forward — the dominant cost of the ~35s/seed marginal-slope pre-warm and
        // the bernoulli-MS centers=20 non-finish (#979). The inner solve in
        // `EvalMode::ValueOnly` already produces the converged block β (the only
        // product the pre-warm needs); surface it as `inner_beta_hint` (and into
        // `outer.warm_cache`) with a zero-length gradient and skip the outer
        // gradient assembly. ValueAndGradient / ValueGradientHessian are unchanged.
        if matches!(order, OuterEvalOrder::Value) {
            return match outerobjectivegradienthessian_labeled(
                family,
                specs,
                &outer_options,
                &label_layout,
                rho,
                warm_ref,
                &rho_prior,
                EvalMode::ValueOnly,
            ) {
                Ok(eval) if eval.inner_converged && eval.objective.is_finite() => {
                    let inner_beta_hint = Some(Array1::from_iter(
                        eval.warm_start
                            .block_beta
                            .iter()
                            .flat_map(|beta| beta.iter().copied()),
                    ));
                    outer.warm_cache = Some(eval.warm_start);
                    outer.last_error = None;
                    Ok(OuterEval {
                        cost: eval.objective,
                        gradient: Array1::zeros(rho.len()),
                        hessian: crate::solver::outer_strategy::HessianResult::Unavailable,
                        inner_beta_hint,
                    })
                }
                Ok(eval) => {
                    outer.warm_cache = Some(eval.warm_start);
                    outer.last_error = Some(
                        "custom-family value-only inner solve did not converge or objective was non-finite"
                            .to_string(),
                    );
                    Ok(OuterEval::infeasible(rho.len()))
                }
                Err(e) => {
                    outer.last_error = Some(e.clone());
                    Err(EstimationError::RemlOptimizationFailed(e))
                }
            };
        }
        let request_hessian =
            matches!(order, OuterEvalOrder::ValueGradientHessian) && need_outer_hessian;
        let eval_result = match outerobjectivegradienthessian_labeled(
            family,
            specs,
            &outer_options,
            &label_layout,
            rho,
            warm_ref,
            &rho_prior,
            if request_hessian {
                EvalMode::ValueGradientHessian
            } else {
                EvalMode::ValueAndGradient
            },
        ) {
            Ok(eval) if !eval.inner_converged => {
                outer.warm_cache = Some(eval.warm_start.clone());
                outer.last_error = Some("custom-family inner solve did not converge".to_string());
                // Recoverable: this trial rho is infeasible (inner solve did not
                // converge), so the outer optimizer retreats rather than the whole
                // run hard-erroring. When the search ultimately reports
                // `converged == false`, the post-run rung samples the proper
                // posterior (never-fail).
                return Ok(OuterEval::infeasible(rho.len()));
            }
            Ok(eval)
                if eval.objective.is_finite()
                    && eval.gradient.iter().all(|v| v.is_finite())
                    && match &eval.outer_hessian {
                        crate::solver::outer_strategy::HessianResult::Analytic(hessian) => {
                            hessian.iter().all(|v| v.is_finite())
                        }
                        crate::solver::outer_strategy::HessianResult::Operator(op) => {
                            !request_hessian || op.dim() == rho.len()
                        }
                        crate::solver::outer_strategy::HessianResult::Unavailable => {
                            !request_hessian
                        }
                    } =>
            {
                let warm_start = eval.warm_start.clone();
                let gradient_norm = eval
                    .gradient
                    .iter()
                    .map(|value| value * value)
                    .sum::<f64>()
                    .sqrt();
                update_custom_outer_inner_cap_from_warm_start(
                    &outer_options,
                    &warm_start,
                    Some(gradient_norm),
                    &mut outer.initial_gradient_norm,
                );
                outer.warm_cache = Some(warm_start.clone());
                store_persistent_custom_family_warm_start(
                    persistent_warm_start_key.as_deref(),
                    specs,
                    &warm_start,
                );
                outer.last_error = None;
                eval
            }
            Ok(_) => {
                outer.last_error =
                    Some("custom-family outer objective/derivatives became non-finite".to_string());
                // Recoverable (data-driven): the objective/derivatives became
                // non-finite at this trial rho (e.g. separation / near-singular
                // information), so the outer optimizer retreats from this infeasible
                // point rather than the whole run hard-erroring. When the search
                // ultimately reports `converged == false`, the post-run rung samples
                // the proper posterior (never-fail).
                return Ok(OuterEval::infeasible(rho.len()));
            }
            Err(e) => {
                // Genuine eval-error (internal computation failure: linalg error,
                // etc.) — NOT data-driven. Leave as a hard Err even when escalation
                // is armed: a real bug must surface, not be silently sampled over.
                // Only the "did not converge" / "non-finite objective" data-driven
                // paths above convert to infeasible-when-armed.
                outer.last_error = Some(e.clone());
                return Err(EstimationError::RemlOptimizationFailed(e));
            }
        };
        let inner_beta_hint = Some(Array1::from_iter(
            eval_result
                .warm_start
                .block_beta
                .iter()
                .flat_map(|beta| beta.iter().copied()),
        ));
        Ok(OuterEval {
            cost: eval_result.objective,
            gradient: eval_result.gradient,
            hessian: eval_result.outer_hessian,
            inner_beta_hint,
        })
    };

    let mut obj = problem.build_objective_with_screening_proxy(
        CustomOuterState::new(persistent_warm_start.clone()),
        |outer: &mut CustomOuterState, rho: &Array1<f64>| {
            // Always use warm cache when available — the previous inner solution
            // gives a much better starting point. This was previously disabled for
            // exact-Hessian families, forcing every inner solve to start from
            // scratch (5-10 Newton steps instead of 1-2 with warm start).
            let warm_ref = screened_outer_warm_start(outer.warm_cache.as_ref(), rho);
            match outerobjectivegradienthessian_labeled(
                family,
                specs,
                &outer_options,
                &label_layout,
                rho,
                warm_ref,
                &rho_prior,
                EvalMode::ValueOnly,
            ) {
                Ok(eval) if eval.inner_converged && eval.objective.is_finite() => {
                    outer.warm_cache = Some(eval.warm_start);
                    outer.last_error = None;
                    Ok(eval.objective)
                }
                Ok(eval) => {
                    outer.warm_cache = Some(eval.warm_start);
                    outer.last_error = Some(
                        "custom-family value-only inner solve did not converge or objective was non-finite"
                            .to_string(),
                    );
                    // Recoverable (data-driven): this value-only probe is the
                    // line-search cost the outer optimizer calls most often. A
                    // non-converged inner solve / non-finite objective at this trial
                    // rho means the point is infeasible — return an infinite cost so
                    // the line search retreats, rather than hard-erroring out of
                    // `problem.run` and bypassing the post-run escalation (sampling)
                    // rung. When the search reports `converged == false` the never-fail
                    // rung samples the proper posterior.
                    Ok(f64::INFINITY)
                }
                Err(e) => {
                    // Genuine eval-error (internal computation failure) — NOT
                    // data-driven. Leave as a hard Err even when escalation is armed
                    // so a real bug surfaces instead of being silently sampled over.
                    outer.last_error = Some(e.clone());
                    Err(EstimationError::RemlOptimizationFailed(e))
                }
            }
        },
        |outer: &mut CustomOuterState, rho: &Array1<f64>| {
            eval_outer(
                outer,
                rho,
                if need_outer_hessian {
                    OuterEvalOrder::ValueGradientHessian
                } else {
                    OuterEvalOrder::ValueAndGradient
                },
            )
        },
        |outer: &mut CustomOuterState, rho: &Array1<f64>, order: OuterEvalOrder| {
            eval_outer(outer, rho, order)
        },
        Some(|outer: &mut CustomOuterState| {
            outer.reset();
        }),
        Some(|outer: &mut CustomOuterState, rho: &Array1<f64>| {
            if label_layout.has_tied_coordinates() {
                return Err(EstimationError::RemlOptimizationFailed(
                    "custom-family EFS is not available for tied coefficient-group precision labels"
                        .to_string(),
                ));
            }
            let warm_ref = screened_outer_warm_start(outer.warm_cache.as_ref(), rho);
            match outerobjectiveefs(
                family,
                specs,
                &outer_options,
                &label_layout.penalty_counts,
                rho,
                warm_ref,
                rho_prior.clone(),
            ) {
                Ok((eval, warm, true)) => {
                    outer.warm_cache = Some(warm);
                    outer.last_error = None;
                    Ok(eval)
                }
                Ok((_eval, warm, false)) => {
                    outer.warm_cache = Some(warm);
                    outer.last_error =
                        Some("custom-family EFS inner solve did not converge".to_string());
                    // Intentionally LEFT as a hard Err even when escalation is armed.
                    // Unlike the BFGS/value-only paths above, an EFS error does NOT
                    // dead-end the run: it surfaces as a recoverable objective-eval
                    // error at the fixed-point bridge (outer_strategy.rs:2409-2410
                    // `into_objective_error` -> `ObjectiveEvalError::recoverable`),
                    // so the EFS seed is rejected / the FixedPoint run returns Err,
                    // and `run_outer`'s fallback cascade (outer_strategy.rs:5297) routes
                    // to the fixed-point-disabled analytic-gradient BFGS attempt. That
                    // attempt is always present here because custom-family declares an
                    // analytic outer gradient (custom_family.rs:11826), so
                    // `automatic_fallback_attempts` (outer_strategy.rs:1502) adds it.
                    // BFGS then evaluates via `eval_outer` / the value-only cost
                    // closure, both of which now retreat-when-armed, so the run reaches
                    // `Ok(converged == false)` and the post-run sampling rung. No
                    // analogous infeasible sentinel is needed at this site.
                    Err(EstimationError::RemlOptimizationFailed(
                        "custom-family EFS inner solve did not converge".to_string(),
                    ))
                }
                Err(e) => {
                    // Genuine eval-error (internal computation failure) — NOT
                    // data-driven. Hard Err so a real bug surfaces.
                    outer.last_error = Some(e.clone());
                    Err(EstimationError::RemlOptimizationFailed(e))
                }
            }
        }),
        |outer: &mut CustomOuterState, rho: &Array1<f64>| {
            let warm_ref = screened_outer_warm_start(outer.warm_cache.as_ref(), rho);
            match custom_family_seed_screening_proxy_labeled(
                family,
                specs,
                &outer_options,
                &label_layout,
                rho,
                warm_ref,
                &rho_prior,
            ) {
                Ok((score, warm_start, _inner_converged)) if score.is_finite() => {
                    outer.warm_cache = Some(warm_start);
                    outer.last_error = None;
                    Ok(score)
                }
                Ok((score, warm_start, _inner_converged)) => {
                    outer.warm_cache = Some(warm_start);
                    outer.last_error = Some(format!(
                        "custom-family seed-screening proxy produced non-finite score {score}"
                    ));
                    Err(EstimationError::RemlOptimizationFailed(
                        "custom-family seed-screening proxy produced non-finite score".to_string(),
                    ))
                }
                Err(e) => {
                    outer.last_error = Some(e.clone());
                    Err(EstimationError::RemlOptimizationFailed(e))
                }
            }
        },
    )
    .with_seed_inner_state(|outer: &mut CustomOuterState, beta: &Array1<f64>| {
        outer.seed_cached_beta(n_rho, specs, beta)
    });

    let outer_result = problem.run(&mut obj, "custom family");

    // ── Discriminating outer-gradient FD audit (issue #1040) ──
    //
    // The custom-family outer ρ-REML loop is driven by `problem.run` above, with
    // `outerobjectivegradienthessian_labeled` as the θ↦(V,∇V,H) evaluator. When
    // the loop FAILS to certify convergence, central-difference the outer
    // criterion component-by-component against the analytic gradient and report
    // the outer-Hessian spectrum — the single diagnostic that forks a
    // non-terminating outer loop into objective↔gradient DESYNC (analytic ≠ FD ⇒
    // the trust region chases a phantom descent forever) vs weak IDENTIFIABILITY
    // (analytic ≈ FD but a ~0 outer-Hessian eigenvalue ⇒ a flat valley).
    //
    // This audit costs 2·n_rho + 1 extra full outer evals (each a coupled inner
    // solve over all n rows), so it must run ONLY on the pathology it diagnoses,
    // never on a healthy fit: gating it by size alone (the original #1040 gate)
    // taxed EVERY production custom-family fit — for `bernoulli-marginal-slope`
    // at n=1500, n_rho=6 it was ~39% of the wall clock (13 phantom evals) on a
    // fit that converged cleanly with nothing to diagnose (gam#979). A
    // certified-converged outer result has, by definition, no desync to find, so
    // the audit only fires when `problem.run` returned `Err` or a non-converged
    // result — exactly when the #1040 verdict is actionable.
    let outer_needs_audit = match &outer_result {
        Ok(result) => !result.converged,
        Err(_) => true,
    };
    if outer_needs_audit {
        const OUTER_FD_AUDIT_MAX_N: usize = 4_000;
        const OUTER_FD_AUDIT_MAX_RHO_DIM: usize = 32;
        let audit_n = specs.iter().map(|s| s.design.nrows()).max().unwrap_or(0);
        if n_rho >= 1 && n_rho <= OUTER_FD_AUDIT_MAX_RHO_DIM && audit_n <= OUTER_FD_AUDIT_MAX_N {
            log::warn!(
                "[OUTER-FD-AUDIT/custom-family] outer did not certify convergence; running desync/identifiability audit n={audit_n} n_rho={n_rho} need_outer_hessian={need_outer_hessian}"
            );
            let mut eval_at = |rho: &Array1<f64>,
                               mode: EvalMode|
             -> Result<
                (
                    f64,
                    Array1<f64>,
                    crate::solver::outer_strategy::HessianResult,
                ),
                String,
            > {
                let e = outerobjectivegradienthessian_labeled(
                    family,
                    specs,
                    &outer_options,
                    &label_layout,
                    rho,
                    None,
                    &rho_prior,
                    mode,
                )?;
                if !e.inner_converged {
                    return Err("inner solve did not converge at audit rho".to_string());
                }
                Ok((e.objective, e.gradient, e.outer_hessian))
            };
            match crate::solver::outer_strategy::outer_gradient_fd_audit(
                &rho0,
                1e-4,
                |i| format!("rho[{i}]"),
                &mut eval_at,
            ) {
                Ok(audit) => audit.log_verdict("custom-family"),
                Err(e) => log::warn!("[OUTER-FD-AUDIT/custom-family] skipped: {e}"),
            }
        }
    }

    let last_error_detail = obj
        .state
        .last_error
        .as_ref()
        .map(|e| {
            format!(
                " last objective error: {}",
                normalize_outer_eval_error_detail(e)
            )
        })
        .unwrap_or_default();

    // Startup-validation escalation net (gam#860). When the outer optimizer
    // returns `Err` because no candidate seed passed startup validation, the
    // raise is a POST-AUDIT NUMERICAL pathology, not an ill-posed input: by the
    // time we reach the outer solve the structural audits have already passed
    // (the #531-class identifiability audit, the #789B zero-events guard, and
    // the #859 cross-fit alignment all raise BEFORE the solver). So an
    // all-seeds-rejected / non-finite-cost failure HERE is a solver numerical
    // defect (e.g. the #787 kappa-driven penalty-topology dim-mismatch) on a
    // structurally-well-posed design — exactly the regime the never-fail
    // posterior-sampling rung exists for. Route it into the SAME AUTO-ESCALATE
    // the non-convergence path below uses, seeding the sampler at the initial ρ
    // (`rho0`, the bootstrap seed), instead of hard-raising. The carve-out is
    // strict: this only catches the post-audit startup-validation failure, never
    // the structural guards above (they keep raising with their own messages),
    // and the degraded refit below STILL raises if even `rho0` produces a
    // non-finite mode (sampling about NaN would manufacture meaningless
    // infinite-width intervals that masquerade as a fit — see the finite-mode
    // check after the refit). The result carries the existing escalation's
    // degraded / sampled-not-certified flagging so confidence is honest.
    let (rho_star, outer_grad_norm, outer_iters, nonconvergence_escalation, outer_certificate) =
        match outer_result {
            Ok(outer_result) => {
                // Geometry-driven terminal escalation. When the outer smoothing
                // optimizer cannot certify convergence, the objective is always
                // *proper* (Jeffreys/PC term unconditionally armed), so a
                // non-convergence here is a geometry signal (indefinite / non-smooth
                // LAML landscape that stalled Strong-Wolfe) — not a reason to fail.
                // Instead we AUTO-ESCALATE to sampling the proper posterior about the
                // best mode the inner solve reached (the never-fail bottom rung; see
                // `hmc::sample_gaussian_mode_posterior`). The fast Arc/EFS path is
                // untouched: this branch is only reached after the optimizer reports
                // non-convergence, so nice landscapes never pay any sampling cost.
                let nonconvergence_escalation = !outer_result.converged;
                if nonconvergence_escalation {
                    log::info!(
                        "[robust] outer smoothing did not certify convergence (plan={} iters={} |g|={}); \
                     AUTO-ESCALATE to never-fail posterior sampling about the best mode",
                        outer_result.plan_used,
                        outer_result.iterations,
                        outer_result.final_grad_norm_report(),
                    );
                }
                (
                    outer_result.rho,
                    outer_result.final_grad_norm,
                    outer_result.iterations,
                    nonconvergence_escalation,
                    outer_result.criterion_certificate,
                )
            }
            Err(e) if outer_startup_failure_is_escalatable(&e) => {
                log::warn!(
                    "[robust] outer smoothing raised at startup validation on a structurally-audited \
                 design (post-audit numerical pathology, gam#860): {e}.{last_error_detail} \
                 AUTO-ESCALATE to never-fail posterior sampling about the initial ρ seed; the \
                 degraded refit below still raises if even the seed produces a non-finite mode.",
                );
                (rho0.clone(), None, 0, true, None)
            }
            Err(e) => {
                return Err(format!(
                "outer smoothing optimization failed after exhausting strategy fallbacks: {e}.{last_error_detail}"
            )
            .into());
            }
        };
    screening_cap.store(0, Ordering::Relaxed);

    let per_block = split_labeled_log_lambdas(&rho_star, &label_layout)?;
    let final_seed = obj.state.warm_cache.clone();
    let mut final_options = options.clone();
    final_options.outer_inner_max_iterations = None;
    let mut inner = inner_blockwise_fit(
        family,
        specs,
        &per_block,
        &final_options,
        final_seed.as_ref(),
    )
    .map_err(|e| {
        format!(
            "outer smoothing optimization failed during final inner refit: \
                     {e}.{last_error_detail}"
        )
    })?;
    if !inner.converged && !nonconvergence_escalation {
        return Err(CustomFamilyError::Optimization {
            context: "fit_custom_family final inner refit",
            reason: format!(
                "outer smoothing optimization final inner refit did not converge after {} cycles.{}",
                inner.cycles, last_error_detail
            ),
        });
    }
    if !inner.converged && nonconvergence_escalation {
        // The mode the inner solve reached is still the seed for the proper
        // posterior; a marginal inner non-convergence only widens the sampled
        // intervals (honest, not wrong). Proceed to assemble + sample.
        log::info!(
            "[robust] final inner refit did not fully converge ({} cycles) under escalation; \
             sampling the proper posterior about the reached mode",
            inner.cycles,
        );
    }
    // Finite-mode carve-out for the escalation net (gam#860). The never-fail
    // rung samples a Gaussian posterior ABOUT the reached mode; that is honest
    // only when the mode is finite (a non-converged-but-finite mode just widens
    // the sampled intervals). If the refit produced a NON-FINITE β — e.g. the
    // degraded startup-validation fallback (`rho0`) still lands on garbage —
    // sampling about NaN would manufacture meaningless infinite-width intervals
    // that masquerade as a fit, so KEEP the hard raise with a clear message
    // rather than escalate. (On the certified path β is finite by construction,
    // so this guard only ever fires on a genuinely broken escalation seed.)
    if nonconvergence_escalation
        && inner
            .block_states
            .iter()
            .any(|state| state.beta.iter().any(|value| !value.is_finite()))
    {
        return Err(CustomFamilyError::Optimization {
            context: "fit_custom_family escalation finite-mode check",
            reason: format!(
                "outer smoothing escalation cannot sample a posterior: the refit mode is \
                 non-finite (β contains NaN/inf), so there is no valid mode to sample about; \
                 this is an ill-posed problem, not a recoverable numerical non-convergence.{}",
                last_error_detail
            ),
        });
    }
    let final_warm_start = constrained_warm_start_from_inner(&rho_star, &inner);
    store_persistent_custom_family_warm_start(
        persistent_warm_start_key.as_deref(),
        specs,
        &final_warm_start,
    );
    refresh_all_block_etas(family, specs, &mut inner.block_states).map_err(|e| {
        format!(
            "outer smoothing optimization failed during final eta refresh: \
             {e}.{last_error_detail}"
        )
    })?;
    let mut covariance_conditional =
        compute_joint_covariance_required(family, specs, &inner.block_states, &per_block, options)?;

    let geometry = compute_joint_geometry(family, specs, &inner.block_states, &per_block).map_err(
        |reason| CustomFamilyError::Optimization {
            context: "fit_custom_family joint geometry",
            reason,
        },
    )?;
    let penalized_objective = inner_penalized_objective(
        &inner,
        include_exact_newton_logdet_h(family, options),
        include_exact_newton_logdet_s(family, options),
        "custom-family fit final outer refit",
    )
    .map_err(|reason| CustomFamilyError::Optimization {
        context: "fit_custom_family penalized objective",
        reason,
    })?;
    // Never-fail terminal rung. Under escalation, sample the proper posterior
    // `N(β̂, H⁻¹)` whose precision `H` is the SAME penalized (Jeffreys-augmented)
    // joint Hessian the inner solve produced at the reached mode `β̂`, and report
    // its honest covariance in place of the optimizer-conditional one. Both `H`
    // and `β̂` are in the reduced (canonical) coordinate space here; the joint
    // lift below (`lift_fit_geometry_to_raw`) carries the sampled covariance back
    // to raw space exactly like the conditional covariance it replaces.
    //
    // Sampling a multivariate normal cannot dead-end: `sample_gaussian_mode_posterior`
    // jitters and Cholesky-factors `H`, so a marginally indefinite boundary
    // Hessian only widens the intervals. If that structural factorization is
    // genuinely impossible (e.g. a non-PSD precision after symmetrization) the
    // sampler returns `Err`; rather than re-introducing the dead-end we then keep
    // the optimizer-conditional covariance (a finite point with its existing SEs)
    // and still return a fit — never an `Err` for non-convergence.
    if nonconvergence_escalation {
        if let Some(geom) = geometry.as_ref() {
            let joint_mode: Array1<f64> = {
                let mut mode = Vec::new();
                for state in &inner.block_states {
                    mode.extend(state.beta.iter().copied());
                }
                Array1::from(mode)
            };
            let precision = geom.penalized_hessian.as_array();
            if joint_mode.len() == precision.nrows()
                && precision.nrows() == precision.ncols()
                && joint_mode.iter().all(|v| v.is_finite())
            {
                let sampling_config =
                    crate::inference::hmc::NutsConfig::for_dimension(joint_mode.len());
                match crate::inference::hmc::sample_gaussian_mode_posterior(
                    joint_mode.view(),
                    precision.view(),
                    &sampling_config,
                ) {
                    Ok(posterior) => {
                        let dim = joint_mode.len();
                        let n = posterior.samples.nrows();
                        if n > 1 {
                            // Sample posterior covariance about the posterior mean
                            // (honest intervals; not the Laplace inverse-Hessian).
                            let mean = &posterior.posterior_mean;
                            let mut cov = Array2::<f64>::zeros((dim, dim));
                            for row in posterior.samples.rows() {
                                let centered = &row.to_owned() - mean;
                                for a in 0..dim {
                                    for b in 0..dim {
                                        cov[[a, b]] += centered[a] * centered[b];
                                    }
                                }
                            }
                            cov.mapv_inplace(|v| v / (n as f64 - 1.0));
                            // DIAGNOSTIC GUARD (no false-confident intervals).
                            // The sampler NEVER fails, so without checking its
                            // mixing diagnostics a divergent (R̂ ≫ 1) / near-zero-
                            // ESS draw would be reported as an "honest" covariance.
                            // That is especially dangerous here: the seed `H` is
                            // the Jeffreys-AUGMENTED precision evaluated at β̂, which
                            // may be NON-converged on a flat (unidentified) joint
                            // direction — so a poorly-mixed chain can report a
                            // FINITE, NARROW interval around an arbitrary point on
                            // that flat direction (the prior's interval), masquer-
                            // ading as data-driven. We therefore only accept the
                            // sampled covariance as honest when the chain actually
                            // mixed; otherwise we INFLATE it to reflect the non-
                            // convergence and flag it low-confidence rather than
                            // silently reporting a Jeffreys-narrowed interval.
                            //
                            // R̂ ≤ 1.05 is the standard "mixed" gate (stricter than
                            // the 1.1 used for a coarse converged/not flag, because
                            // this covariance is reported as honest uncertainty).
                            // The ESS floor scales with dimension (≥ 10 effective
                            // draws per parameter, absolute floor 50) so a chain
                            // that produced essentially no independent information
                            // about the posterior is caught independent of model
                            // size.
                            const RHAT_MIXED_MAX: f64 = 1.05;
                            let ess_floor = (10.0 * dim as f64).max(50.0);
                            let rhat = posterior.rhat;
                            let ess = posterior.ess;
                            let diagnostics_ok = rhat.is_finite()
                                && ess.is_finite()
                                && rhat <= RHAT_MIXED_MAX
                                && ess >= ess_floor;
                            if diagnostics_ok {
                                log::info!(
                                    "[robust] never-fail posterior sampling mixed: dim={dim} \
                                     draws={n} rhat={rhat:.3} ess={ess:.0}; reporting sampled \
                                     covariance as honest intervals",
                                );
                                covariance_conditional = Some(cov);
                            } else {
                                // Non-converged: do NOT report the narrow sampled
                                // covariance as data-driven. Inflate it so the
                                // reported uncertainty reflects the failure to
                                // resolve the posterior — widen by the R̂ excess (a
                                // divergent chain widens hard) and an ESS-deficit
                                // factor (too few independent draws ⇒ the sample
                                // covariance is itself unreliable / too narrow). The
                                // result is a clearly-flagged LOW-CONFIDENCE summary,
                                // never an artificially tight interval, and we still
                                // return a fit (the never-fail guarantee stands).
                                let rhat_factor = if rhat.is_finite() {
                                    rhat.max(1.0)
                                } else {
                                    // R̂ unestimable (too few chains/samples) ⇒
                                    // treat as maximally unresolved.
                                    RHAT_MIXED_MAX
                                };
                                let ess_factor = if ess.is_finite() && ess > 0.0 {
                                    (ess_floor / ess).sqrt().max(1.0)
                                } else {
                                    ess_floor.sqrt()
                                };
                                let inflation = (rhat_factor * rhat_factor) * ess_factor;
                                cov.mapv_inplace(|v| v * inflation);
                                log::warn!(
                                    "[robust] never-fail posterior sampling DID NOT MIX: dim={dim} \
                                     draws={n} rhat={rhat:.3} (>{RHAT_MIXED_MAX}) ess={ess:.0} \
                                     (<{ess_floor:.0}); reporting LOW-CONFIDENCE inflated covariance \
                                     (x{inflation:.2}) instead of a possibly false-confident \
                                     Jeffreys-narrowed interval (intervals are prior-dominated on \
                                     any unidentified joint direction, NOT data-driven)",
                                );
                                covariance_conditional = Some(cov);
                            }
                        }
                    }
                    Err(reason) => {
                        log::warn!(
                            "[robust] never-fail posterior sampling could not factor the precision \
                             ({reason}); retaining optimizer-conditional covariance (still no dead-end)",
                        );
                    }
                }
            }
        }
    }
    let rho_star_physical = expand_labeled_log_lambdas(&rho_star, &label_layout)?;
    let outer_converged = !nonconvergence_escalation;
    assemble_custom_family_fit_result(
        inner,
        BlockwiseFitAssembly {
            rho_physical: rho_star_physical,
            covariance_conditional,
            geometry,
            canonical: Some(&canonical),
            result_specs: raw_specs,
            penalized_objective,
            outer_iterations: outer_iters,
            outer_gradient_norm: outer_grad_norm,
            criterion_certificate: outer_certificate,
            outer_converged,
            context: "fit_custom_family result assembly",
        },
    )
}


pub(crate) fn fit_custom_family_fixed_log_lambdas<
    F: CustomFamily + Clone + Send + Sync + 'static,
>(
    family: &F,
    specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
    warm_start: Option<&CustomFamilyWarmStart>,
    outer_iterations: usize,
    outer_gradient_norm: Option<f64>,
    outer_converged: bool,
) -> Result<crate::solver::estimate::UnifiedFitResult, CustomFamilyError> {
    let penalty_counts = validate_blockspecs(specs)?;
    let rho = flatten_log_lambdas(specs);
    let per_block = split_log_lambdas(&rho, &penalty_counts)?;
    let mut inner = inner_blockwise_fit(
        family,
        specs,
        &per_block,
        options,
        warm_start.map(|warm| &warm.inner),
    )?;
    if !inner.converged {
        return Err(CustomFamilyError::Optimization {
            context: "fit_custom_family_fixed_log_lambdas inner solve",
            reason: format!(
                "fixed-log-lambda inner solve did not converge after {} cycles",
                inner.cycles
            ),
        });
    }
    refresh_all_block_etas(family, specs, &mut inner.block_states)?;
    let covariance_conditional =
        compute_joint_covariance_required(family, specs, &inner.block_states, &per_block, options)?;
    let geometry = compute_joint_geometry(family, specs, &inner.block_states, &per_block).map_err(
        |reason| CustomFamilyError::Optimization {
            context: "fit_custom_family_fixed_log_lambdas joint geometry",
            reason,
        },
    )?;
    let penalized_objective = inner_penalized_objective(
        &inner,
        include_exact_newton_logdet_h(family, options),
        include_exact_newton_logdet_s(family, options),
        "custom-family fixed-log-lambda fit",
    )
    .map_err(|reason| CustomFamilyError::Optimization {
        context: "fit_custom_family_fixed_log_lambdas penalized objective",
        reason,
    })?;
    assemble_custom_family_fit_result(
        inner,
        BlockwiseFitAssembly {
            rho_physical: rho,
            covariance_conditional,
            geometry,
            canonical: None,
            result_specs: specs,
            penalized_objective,
            outer_iterations,
            outer_gradient_norm,
            criterion_certificate: None,
            outer_converged,
            context: "fit_custom_family_fixed_log_lambdas result assembly",
        },
    )
}


pub(crate) fn fit_custom_family_fixed_log_lambda_warm_start<
    F: CustomFamily + Clone + Send + Sync + 'static,
>(
    family: &F,
    specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
) -> Result<(Vec<Array1<f64>>, bool, usize), CustomFamilyError> {
    // Pre-fit identifiability gate. Mirrors the outer-fit gate so
    // warm-start callers (e.g. the survival marginal-slope rigid pilot
    // at survival_marginal_slope.rs ~18078) fail in milliseconds on
    // rank-deficient joint designs instead of spending minutes inside
    // a singular penalised Newton inner system.
    //
    // We deliberately do NOT call `canonicalize_for_identifiability`
    // here: blockwise families capture their per-block designs at
    // construction time (e.g. SurvivalMarginalSlopeFamily holds
    // `self.marginal_design` and `self.logslope_design` at raw width)
    // and their `evaluate*` paths assert on those raw widths when
    // assembling per-row Hessian contributions. Substituting a
    // column-reduced spec under that family would produce a runtime
    // shape mismatch in the family's syr_row_into / row_outer_into
    // calls, masking the audit's diagnostic with a panic later in the
    // pipeline.
    //
    // The principled construction-time orthogonalisation lives in
    // `crate::families::identifiability_compiler` (and the per-family
    // `*_identifiability.rs` modules). Once Phase 4b threads those
    // compiled operators through the family construction sites, the
    // raw joint design will already be rank-clean on entry and this
    // gate becomes a defensive check.
    let audit =
        crate::solver::identifiability_audit::audit_identifiability(specs).map_err(|reason| {
            CustomFamilyError::DimensionMismatch {
                reason: format!(
                    "fit_custom_family_fixed_log_lambda_warm_start identifiability audit failed: {reason}"
                ),
            }
        })?;
    if audit.fatal {
        return Err(CustomFamilyError::Optimization {
            context: "fit_custom_family_fixed_log_lambda_warm_start identifiability audit",
            reason: format!(
                "fatal pre-fit identifiability audit: {summary}",
                summary = audit.summary
            ),
        });
    }
    let penalty_counts = validate_blockspecs(specs)?;
    let rho = flatten_log_lambdas(specs);
    let per_block = split_log_lambdas(&rho, &penalty_counts)?;
    let inner = inner_blockwise_fit(family, specs, &per_block, options, None)?;
    let block_beta: Vec<Array1<f64>> = inner
        .block_states
        .iter()
        .map(|state| state.beta.clone())
        .collect();
    if !block_beta
        .iter()
        .flat_map(|beta| beta.iter())
        .all(|value| value.is_finite())
    {
        return Err(CustomFamilyError::Optimization {
            context: "fit_custom_family_fixed_log_lambda_warm_start",
            reason: "fixed-log-lambda warm start produced non-finite coefficients".to_string(),
        });
    }
    Ok((block_beta, inner.converged, inner.cycles))
}


#[cfg(test)]
mod test_support {
    use super::*;
    use ndarray::{Array1, Array2};

    pub(crate) fn outerobjectivegradienthessian<F: CustomFamily + Clone + Send + Sync + 'static>(
        family: &F,
        specs: &[ParameterBlockSpec],
        options: &BlockwiseFitOptions,
        penalty_counts: &[usize],
        rho: &Array1<f64>,
        warm_start: Option<&ConstrainedWarmStart>,
        eval_mode: EvalMode,
    ) -> Result<(f64, Array1<f64>, Option<Array2<f64>>, ConstrainedWarmStart), String> {
        let result = super::outerobjectivegradienthessian_internal(
            family,
            specs,
            options,
            penalty_counts,
            rho,
            warm_start,
            crate::types::RhoPrior::Flat,
            eval_mode,
        )?;
        Ok((
            result.objective,
            result.gradient,
            result.outer_hessian.materialize_dense()?,
            result.warm_start,
        ))
    }
}
