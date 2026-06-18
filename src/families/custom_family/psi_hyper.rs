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
            let a_penalty_quadratic = 0.5 * beta_block.dot(&s_psi_beta_local);
            let a = psi_terms.objective_psi + a_penalty_quadratic;
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
        pub(crate) block: usize,
        pub(crate) local: usize,
        pub(crate) start: usize,
        pub(crate) end: usize,
        pub(crate) s_psi_local: Array2<f64>,
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

    let derivative_blocks = Arc::clone(&derivative_blocks);

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
                    let block_drift: Arc<dyn HyperOperator> = Arc::new(BlockLocalDrift {
                        local: s_local.clone(),
                        start: cache_i.start,
                        end: cache_i.end,
                        total_dim: total,
                    });
                    b_operator = Some(match b_operator.take() {
                        Some(existing) => {
                            let existing_arc: Arc<dyn HyperOperator> = Arc::from(existing);
                            Box::new(CompositeHyperOperator {
                                dense: None,
                                operators: vec![existing_arc, block_drift],
                                dim_hint: total,
                            }) as Box<dyn HyperOperator>
                        }
                        None => Box::new(BlockLocalDrift {
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

pub(crate) fn evaluate_custom_family_hyper_internal<
    F: CustomFamily + Clone + Send + Sync + 'static,
>(
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

pub(crate) fn evaluate_custom_family_hyper_internal_shared<
    F: CustomFamily + Clone + Send + Sync + 'static,
>(
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

        let robust_jeffreys_hphi =
            custom_family_outer_jeffreys_hphi(family, &inner.block_states, specs, &ranges)?;
        let has_configured_rho_prior = !matches!(rho_prior, crate::types::RhoPrior::Flat);
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
                derivative_blocks.as_ref(),
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
                        crate::types::RhoPrior::Flat,
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
                        outer_hessian: crate::solver::rho_optimizer::HessianResult::Unavailable,
                        warm_start: value_only.warm_start,
                        inner_converged: inner.converged,
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
                derivative_blocks.as_ref(),
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
                        outer_hessian: crate::solver::rho_optimizer::HessianResult::Unavailable,
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
        custom_family_outer_jeffreys_hphi_drift_batched(
            family,
            &inner.block_states,
            specs,
            &ranges,
        )?,
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

pub(crate) fn joint_hyper_options_for_outer_tolerance(
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
    derivative_blocks: SharedDerivativeBlocks,
    warm_start: Option<&ConstrainedWarmStart>,
) -> Result<
    (
        crate::solver::rho_optimizer::EfsEval,
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
