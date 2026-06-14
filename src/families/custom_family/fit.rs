//! The public fit entry points (`fit_custom_family`,
//! `fit_custom_family_with_rho_prior`, fixed-lambda variants), result assembly +
//! output-channel wiring, the raw-coordinate lift, and the effective-df-floor
//! rho-bound machinery.

use super::*;

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
pub(crate) fn lift_block_states_to_raw(
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
pub(crate) fn lift_fit_geometry_to_raw(
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

pub(crate) struct BlockwiseFitAssembly<'a> {
    pub(crate) rho_physical: Array1<f64>,
    pub(crate) covariance_conditional: Option<Array2<f64>>,
    pub(crate) geometry: Option<FitGeometry>,
    pub(crate) canonical: Option<&'a crate::solver::identifiability_canonical::CanonicalSpecs>,
    pub(crate) result_specs: &'a [ParameterBlockSpec],
    pub(crate) penalized_objective: f64,
    pub(crate) outer_iterations: usize,
    pub(crate) outer_gradient_norm: Option<f64>,
    pub(crate) criterion_certificate: Option<crate::solver::outer_strategy::CriterionCertificate>,
    pub(crate) outer_converged: bool,
    pub(crate) context: &'static str,
}

pub(crate) fn assemble_custom_family_fit_result(
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
pub(crate) fn wire_output_channels<F: CustomFamily + ?Sized>(
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
pub(crate) fn outer_startup_failure_is_escalatable(err: &EstimationError) -> bool {
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
pub(crate) const EFFECTIVE_DF_FLOOR: f64 = 1.0;

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
pub(crate) fn unit_weight_term_edf(gammas: &[f64], rho: f64) -> f64 {
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
pub(crate) fn design_penalty_range_gammas(
    design: &DesignMatrix,
    penalty: &PenaltyMatrix,
) -> Option<Vec<f64>> {
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
pub(crate) fn effective_df_floor_rho_upper_bounds(
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
                    // Adapt the inner-cycle cap from THIS probe's converged
                    // cost, exactly as the value+gradient main eval does below.
                    // Value-only line-search probes are the MOST FREQUENT outer
                    // call (several per outer iteration), and omitting the cap
                    // update here left every probe running the full
                    // `inner_max_cycles` (1200) budget even after a warm-started
                    // solve converges in a handful of cycles — the dominant
                    // runtime multiplier on a large joint design (the multinomial
                    // smooth-by-factor >360s cliff). `gradient_norm = None`: a
                    // value-only probe has no gradient, so the cap is driven
                    // purely by the converged cycle count (the gradient-norm
                    // near-optimum uncapping is handled by the main eval).
                    update_custom_outer_inner_cap_from_warm_start(
                        &outer_options,
                        &eval.warm_start,
                        None,
                        &mut outer.initial_gradient_norm,
                    );
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
        pub(crate) const OUTER_FD_AUDIT_MAX_N: usize = 4_000;
        pub(crate) const OUTER_FD_AUDIT_MAX_RHO_DIM: usize = 32;
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
                            pub(crate) const RHAT_MIXED_MAX: f64 = 1.05;
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
