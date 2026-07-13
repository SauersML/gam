//! The public fit entry points (`fit_custom_family`,
//! `fit_custom_family_with_rho_prior`, fixed-lambda variants), result assembly +
//! output-channel wiring, the raw-coordinate lift, and the effective-df-floor
//! rho-bound machinery.

use super::*;

pub fn fit_custom_family<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
) -> Result<gam_solve::model_types::UnifiedFitResult, CustomFamilyError> {
    fit_custom_family_with_rho_prior(family, specs, options, gam_problem::RhoPrior::Flat)
}

/// Lift reduced-space `ParameterBlockState`s back to the raw block
/// dimensions described by `canonical.gauge`. Each block's
/// `beta` becomes `T_i · θ_i` (selection-T zeros dropped raw entries);
/// `eta = design · beta` is invariant under the transform, so the
/// reduced-space `eta` field carries through unchanged.
pub(crate) fn lift_block_states_to_raw(
    canonical: &gam_identifiability::canonical::CanonicalSpecs,
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

/// Re-run the unified identifiability audit at the converged raw-coordinate
/// state when a family exposes dynamic primary scalars. Any change from the
/// pilot verdict invalidates the gauge used by the solve, so result assembly
/// fails closed instead of publishing a locally unidentified or over-reduced
/// fit.
fn audit_converged_identifiability<F: CustomFamily + ?Sized>(
    family: &F,
    raw_specs: &[ParameterBlockSpec],
    canonical: &gam_identifiability::canonical::CanonicalSpecs,
    reduced_states: &[ParameterBlockState],
    outer_iter: usize,
) -> Result<(), CustomFamilyError> {
    let raw_states = lift_block_states_to_raw(canonical, reduced_states.to_vec());
    let Some(family_scalars) = family
        .current_identifiability_family_scalars(&raw_states)
        .map_err(|reason| CustomFamilyError::Optimization {
            context: "converged identifiability scalars",
            reason,
        })?
    else {
        return Ok(());
    };
    let beta_current: Vec<f64> = raw_states
        .iter()
        .flat_map(|state| state.beta.iter().copied())
        .collect();
    let beta_pilot = vec![0.0; beta_current.len()];
    let drift = gam_identifiability::audit::maybe_log_audit_drift(
        raw_specs,
        &canonical.audit,
        &beta_pilot,
        &beta_current,
        Some(&family_scalars),
        outer_iter,
        1,
        family.identifiability_probit_frailty_scale(),
    )
    .map_err(|error| CustomFamilyError::Optimization {
        context: "converged identifiability audit",
        reason: error.to_string(),
    })?
    .ok_or_else(|| CustomFamilyError::Optimization {
        context: "converged identifiability audit",
        reason: "period-one converged audit did not run".to_string(),
    })?;
    if drift.current_rank != drift.pilot_rank
        || drift.current_fatal != drift.pilot_fatal
        || !drift.newly_dropped.is_empty()
        || !drift.recovered.is_empty()
    {
        return Err(CustomFamilyError::Optimization {
            context: "converged identifiability audit",
            reason: format!(
                "identifiability verdict changed after convergence: pilot_rank={} current_rank={} pilot_fatal={} current_fatal={} newly_dropped={} recovered={}",
                drift.pilot_rank,
                drift.current_rank,
                drift.pilot_fatal,
                drift.current_fatal,
                drift.newly_dropped.len(),
                drift.recovered.len(),
            ),
        });
    }
    Ok(())
}

/// Lift a reduced-space conditional covariance and retain the exact active
/// precision frame used by the solver.
///
/// Covariance is a contravariant coefficient uncertainty and therefore pushes
/// forward as `T Σθ Tᵀ`. Precision is a quadratic form on the active tangent
/// space: it must remain `Hθ`, accompanied by the affine gauge
/// `βraw = T θ + a`. A rectangular `T` has no full raw-coordinate inverse, so
/// sandwiching `Hθ` as if it were covariance manufactures a rank-deficient
/// matrix that is not a precision. Saved ALO pulls raw row Jacobians back as
/// `Jθ = Jraw T` and solves this retained `Hθ` exactly.
pub(crate) fn lift_fit_geometry_to_raw(
    canonical: &gam_identifiability::canonical::CanonicalSpecs,
    covariance_conditional: Option<Array2<f64>>,
    geometry: Option<FitGeometry>,
) -> Result<(Option<Array2<f64>>, Option<FitGeometry>), CustomFamilyError> {
    let lifted_cov = covariance_conditional.map(|c| canonical.gauge.lift_covariance(&c));
    let lifted_geom = lift_fit_geometry_through_gauge(&canonical.gauge, geometry)?;
    Ok((lifted_cov, lifted_geom))
}

pub(crate) fn lift_fit_geometry_through_gauge(
    raw_from_geometry: &Gauge,
    geometry: Option<FitGeometry>,
) -> Result<Option<FitGeometry>, CustomFamilyError> {
    geometry
        .map(|mut geometry| {
            geometry.coefficient_gauge = geometry
                .coefficient_gauge
                .left_compose(raw_from_geometry)
                .map_err(|reason| CustomFamilyError::InvalidInput {
                    context: "lift_fit_geometry_through_gauge",
                    reason,
                })?;
            Ok::<_, CustomFamilyError>(geometry)
        })
        .transpose()
}

fn fixed_lambda_warm_start_for_reduced_specs<'a>(
    warm_start: Option<&'a CustomFamilyWarmStart>,
    canonical: &gam_identifiability::canonical::CanonicalSpecs,
) -> Option<&'a ConstrainedWarmStart> {
    let warm = warm_start?;
    if !canonical.gauge.is_identity() {
        return None;
    }
    if warm.inner.block_beta.len() != canonical.reduced_specs.len()
        || warm.inner.active_sets.len() != canonical.reduced_specs.len()
    {
        return None;
    }
    let widths_match = warm
        .inner
        .block_beta
        .iter()
        .zip(canonical.reduced_specs.iter())
        .all(|(beta, spec)| beta.len() == spec.design.ncols());
    widths_match.then_some(&warm.inner)
}

pub(crate) struct BlockwiseFitAssembly<'a> {
    pub(crate) rho_physical: Array1<f64>,
    pub(crate) covariance_conditional: Option<Array2<f64>>,
    pub(crate) geometry: Option<FitGeometry>,
    pub(crate) canonical: Option<&'a gam_identifiability::canonical::CanonicalSpecs>,
    pub(crate) result_specs: &'a [ParameterBlockSpec],
    pub(crate) penalized_objective: f64,
    pub(crate) outer_iterations: usize,
    pub(crate) outer_gradient_norm: Option<f64>,
    pub(crate) criterion_certificate: Option<gam_solve::rho_optimizer::OuterCriterionCertificate>,
    pub(crate) outer_converged: bool,
    /// Selected per-component log-smoothing parameters of the full-width JOINT
    /// penalty at ρ* (gam#1587/#561), surfaced on `FitArtifacts` so a
    /// joint-penalized family (the multinomial centered metric) can recover its
    /// converged smoothing. `None` for every per-block-only family.
    pub(crate) joint_log_lambdas: Option<Array1<f64>>,
}

pub(crate) fn assemble_custom_family_fit_result(
    inner: BlockwiseInnerResult,
    assembly: BlockwiseFitAssembly<'_>,
) -> Result<gam_solve::model_types::UnifiedFitResult, CustomFamilyError> {
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
        joint_log_lambdas,
    } = assembly;
    let log_lambdas = rho_physical;
    let lambdas =
        exact_lambdas_from_log_strengths(&log_lambdas, "custom-family fitted log strength")?;
    let (block_states, covariance_conditional, geometry, precomputed_edf) =
        if let Some(canonical) = canonical {
            let precomputed_edf = reduced_blockwise_edf(geometry.as_ref(), canonical, &lambdas);
            let block_states = lift_block_states_to_raw(canonical, inner.block_states);
            let (covariance_conditional, geometry) =
                lift_fit_geometry_to_raw(canonical, covariance_conditional, geometry)?;
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
            joint_log_lambdas,
        },
        result_specs,
    )
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
fn unit_weight_term_edf_at_physical_strength(gammas: &[f64], lambda: f64) -> f64 {
    gammas
        .iter()
        .map(|&gamma| {
            if gamma.is_finite() && gamma > 0.0 {
                1.0 / (1.0 + lambda / gamma)
            } else {
                0.0
            }
        })
        .sum()
}

pub(crate) fn unit_weight_term_edf(gammas: &[f64], rho: f64) -> Result<f64, CustomFamilyError> {
    let lambda = gam_problem::checked_exp_log_strength(rho).map_err(|error| {
        CustomFamilyError::InvalidInput {
            context: "unit-weight structural EDF",
            reason: error.to_string(),
        }
    })?;
    Ok(unit_weight_term_edf_at_physical_strength(gammas, lambda))
}

/// Generalized eigenvalues `γ_j` of the design column Gram `G = XᵀX` against the
/// penalty `S` on `range(S)`, computed structurally (unit weights).
///
/// These are the eigenvalues of the pencil `(UᵀG U, D)` where `S = U D Uᵀ` and
/// the index runs over `range(S)` (the positive eigenvalues `d_j` of `S`),
/// QUOTIENTED by `ker(S)`: with `A = UᵀGU` partitioned into null (`0`) and
/// range (`r`) blocks they are the eigenvalues of the symmetric matrix
///
/// ```text
/// B = D_r^{-1/2} (A_rr − A_r0 A₀₀⁺ A₀r) D_r^{-1/2},
/// ```
///
/// with `D_r = diag(d_j)` over the range and `U` the penalty eigenvectors.
/// The Schur complement is essential whenever `G` couples the penalized range
/// to `ker(S)`: null directions are fitted unpenalized at every λ and absorb
/// the shared curvature, so `A_rr` alone overstates the λ-resistant df. With
/// these `γ_j` the structural effective df obeys the EXACT trace identity
///
/// ```text
/// rank(A₀₀) + Σ_j γ_j/(γ_j + λ) = tr{ G (G + λ S)⁻¹ }   for all λ > 0,
/// ```
///
/// whose λ-dependent part is the returned spectrum.
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
    // Split the penalty eigenbasis into range(S) columns U_r (d_j above the
    // numerical-zero threshold, with inverse square-root weights d_j^{-1/2})
    // and ker(S) columns U_0. Null directions carry no penalty, but they are
    // NOT simply dropped: they are fitted unpenalized for every λ, so any
    // design curvature they share with the range is absorbed by them and must
    // be projected out of the range curvature (Schur complement below).
    let mut range_cols: Vec<usize> = Vec::new();
    let mut inv_sqrt_d: Vec<f64> = Vec::new();
    let mut null_cols: Vec<usize> = Vec::new();
    for (j, &dj) in s_evals.iter().enumerate() {
        if dj <= s_thresh {
            null_cols.push(j);
        } else {
            range_cols.push(j);
            inv_sqrt_d.push(1.0 / dj.sqrt());
        }
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
    let mut b = y.t().dot(&gram).dot(&y);
    if !null_cols.is_empty() {
        // Quotient the null space out of the range curvature. In the penalty
        // eigenbasis, with A = UᵀGU partitioned into null (0) and range (r)
        // blocks, the λ-dependent part of the exact trace identity
        //   tr{ G (G + λS)⁻¹ } = rank(A₀₀) + Σ_j γ_j/(γ_j + λ)
        // has γ_j the eigenvalues of the SCHUR COMPLEMENT
        //   D_r^{-1/2} (A_rr − A_r0 A₀₀⁺ A₀r) D_r^{-1/2},
        // not of A_rr alone: a range direction whose design curvature is
        // shared with ker(S) contributes NO λ-resistant df of its own — the
        // unpenalized null coordinate absorbs that fit at every λ. Keeping
        // only A_rr (the pre-#audit behaviour) overstates the structural edf
        // (S = diag(0,1,1), G coupling coordinates 1↔2 with residual ε gives
        // quotient eigenvalues (ε, 1), not (1+ε, 1)) and mis-places the
        // smoothing-collapse barrier.
        let r0 = null_cols.len();
        let mut u0 = Array2::<f64>::zeros((p, r0));
        for (col, &src) in null_cols.iter().enumerate() {
            let u = s_evecs.column(src);
            for row in 0..p {
                u0[(row, col)] = u[row];
            }
        }
        let g00 = u0.t().dot(&gram).dot(&u0); // r0×r0
        let g_r0 = y.t().dot(&gram).dot(&u0); // r×r0, already D_r^{-1/2}-scaled rows
        // A₀₀⁺ through the null-block eigendecomposition (r0 is small); the
        // pseudo-inverse (not an inverse) because the design need not have
        // full column support on ker(S).
        let mut g00_sym = g00.clone();
        for i in 0..r0 {
            for j in (i + 1)..r0 {
                let avg = 0.5 * (g00_sym[(i, j)] + g00_sym[(j, i)]);
                g00_sym[(i, j)] = avg;
                g00_sym[(j, i)] = avg;
            }
        }
        let (e0, v0) = g00_sym.eigh(Side::Lower).ok()?;
        let tol0 = positive_eigenvalue_threshold(e0.as_slice()?);
        // B ← B − G_r0 · A₀₀⁺ · G_r0ᵀ, accumulated per retained null mode:
        // with w_k = G_r0 v0_k, subtract e0_k⁻¹ · w_k w_kᵀ.
        for k in 0..r0 {
            if e0[k] <= tol0 {
                continue;
            }
            let inv_e = 1.0 / e0[k];
            let w_k = g_r0.dot(&v0.column(k));
            for i in 0..r {
                for j in 0..r {
                    b[(i, j)] -= inv_e * w_k[i] * w_k[j];
                }
            }
        }
    }
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
) -> Result<Array1<f64>, CustomFamilyError> {
    gam_problem::validate_log_strength(ceiling).map_err(|error| {
        CustomFamilyError::InvalidInput {
            context: "effective-DF rho ceiling",
            reason: error.to_string(),
        }
    })?;
    gam_problem::validate_log_strength(-ceiling).map_err(|error| {
        CustomFamilyError::InvalidInput {
            context: "effective-DF rho lower bound",
            reason: error.to_string(),
        }
    })?;
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
            let edf_max = unit_weight_term_edf_at_physical_strength(&gammas, 0.0);
            if !(edf_max > EFFECTIVE_DF_FLOOR) {
                continue;
            }
            // Bisect for ρ* with edf(ρ*) = floor on [−ceiling, ceiling]; edf is
            // monotone decreasing in ρ. If edf at the ceiling still exceeds the
            // floor, the uniform ceiling already retains enough df — keep it.
            if unit_weight_term_edf(&gammas, ceiling)? >= EFFECTIVE_DF_FLOOR {
                continue;
            }
            // If the existing lower side of the box has already smoothed this
            // term below the structural floor, the floor is not enforceable
            // inside the optimizer's admissible domain. Do not manufacture an
            // upper bound numerically indistinguishable from (or below, after
            // the optimizer's strict bound-validation tolerance is applied)
            // the lower bound: that turns a legitimate model into an invalid
            // rho-box before the data likelihood is even evaluated. This case
            // occurs for very weakly scaled range-space directions, including
            // dispersion location-scale smooths whose unit-weight generalized
            // eigenvalues can put the edf=1 crossing just outside the default
            // [-10, 10] rho box.
            if unit_weight_term_edf(&gammas, -ceiling)? <= EFFECTIVE_DF_FLOOR {
                continue;
            }
            let mut lo = -ceiling;
            let mut hi = ceiling;
            for _ in 0..64 {
                let mid = 0.5 * (lo + hi);
                if unit_weight_term_edf(&gammas, mid)? >= EFFECTIVE_DF_FLOOR {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            let rho_star = 0.5 * (lo + hi);
            // Tied coordinates: take the tightest (smallest) bound across terms,
            // so every term sharing this λ retains at least the floor.
            let slot = &mut upper[outer];
            if rho_star > -ceiling + 1e-6 && rho_star < *slot {
                *slot = rho_star;
            }
        }
    }
    Ok(upper)
}

pub fn fit_custom_family_with_rho_prior<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
    rho_prior: gam_problem::RhoPrior,
) -> Result<gam_solve::model_types::UnifiedFitResult, CustomFamilyError> {
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
    // covers all four `solver::fit_orchestration.rs` entry points plus every
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
    let canonical = gam_identifiability::canonical::canonicalize_for_identifiability(raw_specs)?;
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

    // gam#1587: full-width cross-block joint penalties (the reference-symmetric
    // `M⊗S_t` multinomial smoothing penalty). Empty for every other family, so
    // the joint-penalty code paths below are skipped and behaviour is identical.
    // The specs are produced in raw (pre-canonicalisation) stacked coordinates;
    // pull each back through the identifiability gauge `T_full`
    // (`S_red = T_fullᵀ S_raw T_full`) so it acts on the reduced coordinate space
    // the inner solve and outer evaluator run in.
    let reduced_total: usize = specs.iter().map(|s| s.design.ncols()).sum();
    let joint_specs: Vec<gam_problem::JointPenaltySpec> = {
        let raw_specs_joint =
            family
                .joint_penalty_specs()
                .map_err(|reason| CustomFamilyError::Optimization {
                    context: "fit_custom_family joint penalty specs",
                    reason,
                })?;
        let t_full = &canonical.gauge.t_full;
        // The trait contract fixes the coordinates: joint penalties arrive in
        // RAW (pre-canonicalisation) stacked coordinates, so the pullback
        // decision must key on whether the gauge is the identity — NOT on a
        // dimension comparison. When no columns are dropped the raw and
        // reduced totals coincide even though `T_full` can still be a
        // nontrivial rotation, and skipping `TᵀST` there would smooth the
        // wrong quadratic form (a coordinate swap with S = diag(1,2) must
        // become diag(2,1)).
        let gauge_is_identity = t_full.nrows() == t_full.ncols()
            && t_full
                .indexed_iter()
                .all(|((i, j), &v)| v == if i == j { 1.0 } else { 0.0 });
        raw_specs_joint
            .into_iter()
            .map(|spec| {
                if spec.matrix.nrows() != t_full.nrows() {
                    return Err(CustomFamilyError::DimensionMismatch {
                        reason: format!(
                            "joint penalty '{}' has dim {} but the trait contract requires the \
                             raw stacked total {} (pre-canonicalisation coordinates)",
                            spec.label.as_deref().unwrap_or("<unlabeled>"),
                            spec.matrix.nrows(),
                            t_full.nrows(),
                        ),
                    });
                }
                let (pulled, nullspace_dim) = if gauge_is_identity {
                    (spec.matrix, spec.nullspace_dim)
                } else {
                    let pulled = t_full.t().dot(&spec.matrix).dot(t_full);
                    // The gauge changes rank/nullity nontrivially — a dropped
                    // or rotated column can absorb penalized directions or
                    // fold null directions away (reducing diag(1,0) to its
                    // first coordinate has nullity 0, not 1) — so the declared
                    // raw nullity is recomputed on the pulled-back operator
                    // instead of being capped at the reduced total.
                    let (evals, _) = pulled.eigh(Side::Lower).map_err(|e| {
                        CustomFamilyError::Optimization {
                            context: "fit_custom_family joint penalty pullback rank",
                            reason: format!(
                                "eigendecomposition of pulled-back joint penalty '{}' failed: {e}",
                                spec.label.as_deref().unwrap_or("<unlabeled>"),
                            ),
                        }
                    })?;
                    let evals_slice =
                        evals
                            .as_slice()
                            .ok_or_else(|| CustomFamilyError::Optimization {
                                context: "fit_custom_family joint penalty pullback rank",
                                reason: "non-contiguous eigenvalue buffer".to_string(),
                            })?;
                    let thresh = positive_eigenvalue_threshold(evals_slice);
                    let rank = evals.iter().filter(|&&ev| ev > thresh).count();
                    (pulled, reduced_total - rank)
                };
                let out = gam_problem::JointPenaltySpec {
                    label: spec.label,
                    matrix: pulled,
                    initial_log_lambda: spec.initial_log_lambda,
                    nullspace_dim,
                };
                out.validate()
                    .map_err(|e| CustomFamilyError::ConstraintViolation {
                        reason: format!("joint penalty validation failed: {e}"),
                    })?;
                Ok(out)
            })
            .collect::<Result<Vec<_>, CustomFamilyError>>()?
    };

    let label_layout = penalty_label_layout_with_joint(specs, penalty_counts.clone(), joint_specs)?;
    let mut rho0 = label_layout.initial_rho.clone();
    let (persistent_warm_start_key, mut persistent_warm_start) =
        load_persistent_custom_family_warm_start::<F>(family, specs, options, rho0.len());
    // The cross-fit `FitArtifact` transfer (consume/capture below) reuses
    // per-block β/ρ from a structurally-matching prior fit under a descriptor
    // key that deliberately EXCLUDES the response. Per the
    // `persistent_warm_start_fingerprint` contract, reusing β across fits is
    // only admissible for families that opt into persistent warm-starts by
    // providing a likelihood-data fingerprint (which is exactly what makes
    // `persistent_warm_start_key` `Some`). Families that opt out (fingerprint
    // `None` ⇒ key `None`) must cold-start so repeat fits of the same model are
    // bit-reproducible: without this gate a second structurally-identical fit
    // warm-starts off the first and settles on a different point within the
    // inner solve's flat-basin tolerance (gam#1607 cluster 4 — the location-
    // scale engine-vs-reference exact-replay parity), and successive process
    // runs drift as each seeds off the previous run's on-disk artifact.
    let cross_fit_artifact_enabled = persistent_warm_start_key.is_some();

    // Cross-fit warm start: when the exact response-keyed inner cache MISSES
    // (a new fold / row population / reduced width), fall back to the
    // descriptor-indexed FitArtifact store and transfer BOTH the smoothing
    // parameters ρ AND a function-space-projected starting β from a
    // structurally-matching prior fit. The parent stores RAW β; we least-
    // squares project it onto this fold's reduced subspace via the new gauge
    // lift `T_b`, so the transfer survives a differing reduced width (the LOSO
    // p=37 vs p=35 case that the exact-key path skips with "cached inner beta
    // length mismatch"). This is exactness-preserving — a warm (ρ, β) only sets
    // the inner Newton / outer REML starting iterate, which still runs to its
    // KKT/REML certificate — and behavior-neutral on a cold store (no parent ⇒
    // rho0 + cold β unchanged). Any anomaly degrades that block (or the whole
    // transfer) to cold.
    if cross_fit_artifact_enabled && persistent_warm_start.is_none() && !rho0.is_empty() {
        if let Some(warm) = consume_fit_artifact::<F>(
            specs,
            &canonical.gauge,
            &label_layout.physical_to_outer,
            &rho0,
        ) {
            let beta_widths_ok = warm.block_beta.len() == specs.len()
                && warm
                    .block_beta
                    .iter()
                    .zip(specs.iter())
                    .all(|(beta, spec)| beta.len() == spec.design.ncols());
            if warm.rho.len() == rho0.len()
                && warm.rho.iter().all(|v| v.is_finite())
                && beta_widths_ok
            {
                rho0 = warm.rho.clone();
                // Route the projected β through the same inner warm-start
                // channel the exact-key path uses (`CustomOuterState::new`):
                // the inner solve's cold-start path copies per-block β where
                // the reduced width matches and ignores it otherwise.
                persistent_warm_start = Some(warm);
            }
        }
    }

    if rho0.is_empty() {
        let physical_rho0 = expand_labeled_log_lambdas(&rho0, &label_layout)?;
        let per_block = split_labeled_log_lambdas(&rho0, &label_layout)?;
        let mut inner = inner_blockwise_fit(
            family,
            specs,
            &per_block,
            options,
            persistent_warm_start.as_ref(),
        )
        .map_err(|error| CustomFamilyError::Optimization {
            context: "fit_custom_family no-smoothing inner solve",
            reason: format!("{error}; no fit was assembled"),
        })?;
        let warm_start = constrained_warm_start_from_inner(&rho0, &inner);
        store_persistent_custom_family_warm_start(
            persistent_warm_start_key.as_deref(),
            specs,
            &warm_start,
        );
        if !inner.converged {
            return Err(CustomFamilyError::Optimization {
                context: "fit_custom_family no-smoothing inner solve",
                reason: format!(
                    "coefficient optimization did not converge after {} cycles; no fit was \
                     assembled",
                    inner.cycles
                ),
            });
        }
        refresh_all_block_etas(family, specs, &mut inner.block_states)?;
        audit_converged_identifiability(family, raw_specs, &canonical, &inner.block_states, 0)?;
        let covariance_conditional = compute_joint_covariance_required(
            family,
            specs,
            &inner.block_states,
            &per_block,
            options,
        )
        .map_err(|error| CustomFamilyError::Optimization {
            context: "fit_custom_family no-smoothing covariance factorization",
            reason: format!("{error}; no fit was assembled"),
        })?;
        let reml_term = if options.use_remlobjective {
            0.5 * (inner.block_logdet_h - inner.block_logdet_s)
        } else {
            0.0
        };
        let geometry =
            compute_joint_geometry(family, specs, &inner.block_states, &per_block, options)
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
        // Cross-fit FitArtifact capture (Phase 0/1): persist the converged
        // raw-β + ρ under the descriptor-indexed keyspace so a later fold
        // can warm-start its ρ. Best-effort; never affects this fit. Gated on
        // the same opt-in as the consume side (gam#1607) so opt-out families
        // publish nothing and stay bit-reproducible across repeat fits/runs.
        if cross_fit_artifact_enabled {
            capture_fit_artifact::<F>(
                specs,
                &canonical.gauge,
                &warm_start.block_beta,
                &warm_start.rho,
                &label_layout.physical_to_outer,
                penalized_objective,
                true,
            );
        }
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
                outer_converged: true,
                joint_log_lambdas: None,
            },
        );
    }

    // Exact Hessians are primary whenever the assembled family can supply them.
    // If a particular outer step is ill-conditioned, strategy fallback handles
    // the downgrade; we do not suppress second-order capability preemptively
    // based on the presence of a wiggle block. Small iteration budgets still
    // run through this same outer solver and must earn its convergence
    // certificate; they are not a production shortcut to an unoptimized fit.
    use gam_problem::OuterEval;
    use gam_solve::model_types::EstimationError;
    use gam_solve::rho_optimizer::{FallbackPolicy, OuterEvalOrder, OuterProblem};

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
    let bfgs_step_cap = first_order_bfgs_loglambda_step_cap(need_outer_hessian);
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
        .with_rel_cost_tolerance(options.outer_rel_cost_tol)
        .with_max_iter(options.outer_max_iter)
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
            Array1::<f64>::from_elem(n_rho, options.rho_lower_bound),
            effective_df_floor_rho_upper_bounds(specs, &label_layout, n_rho, 10.0)?,
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
    //
    // WARM-START SHORT-CIRCUIT (biobank LOSO perf): the seed-screening cascade
    // exists only to discover a good COLD starting seed when none is supplied —
    // it runs a full inner solve (the ~8s/seed per-row cell-moment exact-cache
    // build) for each of the 5..N cold seeds, ~43s total, purely to RANK them
    // and pick a starting ρ. When a validated warm (ρ, β) is already present —
    // either the exact response-keyed persistent loader hit, or the cross-fit
    // FitArtifact projection fired above (`persistent_warm_start.is_some()`,
    // with `rho0` already replaced by the warm ρ) — that warm ρ IS the
    // near-optimal starting seed the screen would otherwise spend ~43s
    // rediscovering. So we treat a present warm start exactly like a pinned
    // `initial_rho`: leave the screening cap `None`, and the warm ρ flows
    // straight into the BFGS/Newton outer solver.
    //
    // No-result-change: the screen only SELECTS a starting seed; it never
    // alters the converged ρ. The outer optimizer still runs from the warm ρ
    // and must reach its KKT/REML box-constraint stationarity certificate
    // (the iter-0-metric fix `0eeb2d17b` makes a near-optimal warm seed
    // converge in ~1 step), so the certified ρ is unchanged — we only remove
    // the redundant cold-seed exploration the warm start already supersedes.
    //
    // Cold-fit safety: on a cold fit (no persistent hit AND the cross-fit
    // `consume_fit_artifact` returned `None`), `persistent_warm_start` is
    // `None`, so `warm_start_present` is `false` and the FULL multi-seed
    // screen runs unchanged — cold fits keep their multi-seed robustness.
    let warm_start_present = persistent_warm_start.is_some();
    if warm_start_present {
        log::info!(
            "[OUTER] custom family: warm-start present (ρ/β seed already near-optimal); \
             skipping cold seed-screening cascade, proceeding straight to BFGS/Newton certificate"
        );
    }
    let problem = if options.screen_initial_rho && !warm_start_present {
        problem.with_screening_cap(Arc::clone(&screening_cap))
    } else {
        problem
    };
    // Attach an explicit warm-start session when the caller supplied one.
    // This makes the custom-family outer optimizer (BFGS / ARC depending on
    // derivative capabilities) use the same persistent cache infrastructure as
    // standard REML. Ordinary workflow fits leave this empty so refit-heavy CI
    // loops do not pay shared-store lookup/checkpoint/eviction I/O.
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

    // An inner failure at one trial rho makes that trial infeasible, not the
    // entire smoothing problem. Let the outer optimizer retreat and try its
    // remaining certified strategies. If none reaches stationarity, the outer
    // result below is returned as nonconvergence with checkpoint evidence;
    // no fit or posterior approximation is assembled from the trial state.
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
                        hessian: gam_problem::HessianValue::Unavailable,
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
                // Recoverable at the trial level: the outer optimizer may
                // retreat to another rho, but this state can never certify the
                // outer solve or reach result assembly.
                return Ok(OuterEval::infeasible(rho.len()));
            }
            Ok(eval)
                if eval.objective.is_finite()
                    && eval.gradient.iter().all(|v| v.is_finite())
                    && match &eval.outer_hessian {
                        gam_problem::HessianValue::Dense(hessian) => {
                            hessian.iter().all(|v| v.is_finite())
                        }
                        gam_problem::HessianValue::Operator(op) => {
                            !request_hessian || op.dim() == rho.len()
                        }
                        gam_problem::HessianValue::Unavailable => !request_hessian,
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
                // point rather than the whole run hard-erroring. Exhausting
                // those alternatives becomes a terminal nonconvergence error.
                return Ok(OuterEval::infeasible(rho.len()));
            }
            Err(e) => {
                // Genuine evaluator failure (for example, invalid linear
                // algebra) is not a trial-level infeasibility. Surface it as a
                // hard error instead of letting strategy fallback obscure it.
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
        CustomOuterState::new(persistent_warm_start.clone())
            .with_outer_derivative_pilot(family.outer_derivative_pilot_schedule()),
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
                    // the line search retreats. If every candidate remains
                    // infeasible, `problem.run` returns a terminal error and no
                    // fit is assembled.
                    Ok(f64::INFINITY)
                }
                Err(e) => {
                    // Genuine evaluator failure is not data-driven trial
                    // infeasibility, so it remains a hard error.
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
            if !label_layout.supports_direct_physical_efs() {
                return Err(EstimationError::RemlOptimizationFailed(
                    "custom-family EFS requires an identity per-block penalty-coordinate layout with no fixed, tied, or joint penalties"
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
                    // EFS cannot form a valid fixed-point update away from an
                    // inner mode. Returning an error lets the outer strategy
                    // runner try an analytically valid alternative; exhaustion
                    // remains a terminal nonconvergence error.
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
    })
    .with_exact_polish(CustomOuterState::begin_exact_polish);

    let outer_result = problem.run(&mut obj, "custom family");

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

    // SPEC 20: a fit object only ever comes from a certified-converged outer
    // optimization. A non-converged outer result is a typed nonconvergence
    // error carrying its evidence (plan, iterations, gradient norm, and rho
    // checkpoint). A keyed coefficient warm start is persisted when the family
    // supplies the response fingerprint required for safe reuse. Work survives
    // through checkpoint/resume, never by minting a degraded fit. A
    // Laplace/Gaussian approximation centered at a
    // non-mode is not posterior inference, so there is no sampling rung to
    // "escalate" to here: sampling is only ever legitimate about a certified
    // mode, and a certified mode has no nonconvergence to recover from.
    let (rho_star, outer_grad_norm, outer_iters, outer_certificate) = match outer_result {
        Ok(outer_result)
            if outer_result.converged
                && outer_result
                    .criterion_certificate
                    .as_ref()
                    .is_some_and(|certificate| certificate.certifies()) =>
        {
            (
                outer_result.rho,
                outer_result.final_grad_norm,
                outer_result.iterations,
                outer_result.criterion_certificate,
            )
        }
        Ok(outer_result) => {
            if let Some(warm) = obj.state.warm_cache.as_ref() {
                store_persistent_custom_family_warm_start(
                    persistent_warm_start_key.as_deref(),
                    specs,
                    warm,
                );
            }
            let certificate = outer_result
                .criterion_certificate
                .as_ref()
                .map_or_else(|| "missing".to_string(), |value| value.summary());
            return Err(CustomFamilyError::Optimization {
                context: "fit_custom_family outer smoothing",
                reason: format!(
                    "outer smoothing optimization did not certify convergence \
                     (plan={}, iterations={}, |grad|={}, certificate={}, \
                     rho_checkpoint={:?}); no fit was assembled.{}",
                    outer_result.plan_used,
                    outer_result.iterations,
                    outer_result.final_grad_norm_report(),
                    certificate,
                    outer_result.rho.as_slice().unwrap_or(&[]),
                    last_error_detail
                ),
            });
        }
        Err(e) => {
            let rho_checkpoint = obj
                .state
                .warm_cache
                .as_ref()
                .map(|warm| warm.rho.to_vec())
                .unwrap_or_else(|| rho0.to_vec());
            if let Some(warm) = obj.state.warm_cache.as_ref() {
                store_persistent_custom_family_warm_start(
                    persistent_warm_start_key.as_deref(),
                    specs,
                    warm,
                );
            }
            return Err(CustomFamilyError::Optimization {
                context: "fit_custom_family outer smoothing",
                reason: format!(
                    "outer smoothing optimization failed after exhausting strategy fallbacks: \
                     {e}; rho_checkpoint={rho_checkpoint:?}; no fit was assembled.\
                     {last_error_detail}"
                ),
            });
        }
    };
    screening_cap.store(0, Ordering::Relaxed);

    let per_block = split_labeled_log_lambdas(&rho_star, &label_layout)?;
    // Seed the final β̂ refit at ρ* from the outer optimizer's warm cache.
    //
    // When the cache's ρ bit-matches ρ* the seed is passed whole: the inner
    // solve's same-ρ fast path reuses the cached converged mode (logdets,
    // penalty, active constraints) directly.
    //
    // When it does NOT match (the last accepted outer eval sat at a nearby
    // trial ρ, not ρ*), the ρ-specific `cached_inner` is invalid and MUST NOT
    // be reused — but the converged block β at that nearby ρ is still the best
    // available continuation seed for the final refit's coupled joint Newton.
    // Previously the whole seed was dropped to `None` here, forcing the refit
    // to COLD-START from the family-default β. On a stiff two-block
    // location-scale basin that cold start can diverge even though the outer
    // search already certified ρ*: with a `bs='tp', k>=20` scale smooth the
    // refit drove the *mean* block to |β|~10 and aborted with a KKT
    // cert-refusal (`phantom_multiplier_with_well_conditioned_H`), while
    // k=25 — a different, more forgiving basin — converged. Keeping the β
    // continuation (and active sets) seeds the refit at the outer optimum so
    // the coupled Newton opens next to its solution instead of cold (#1561).
    // The inner solve already re-gates cache-mode reuse on its own
    // `warm_start_matches_block_log_lambdas` check, so stripping `cached_inner`
    // here is the belt-and-suspenders guarantee that a mismatched-ρ seed
    // contributes ONLY its β/active-set continuation, never a stale mode.
    let final_seed = obj.state.warm_cache.clone().map(|mut seed| {
        if !warm_start_matches_block_log_lambdas(&seed, &per_block) {
            seed.cached_inner = None;
        }
        seed
    });
    let mut final_options = options.clone();
    final_options.outer_inner_max_iterations = None;
    // gam#1587: the final β̂ refit must apply the same full-width joint penalty
    // at the converged ρ* as every outer eval did, or the reported coefficients
    // (and predictions) would be the UNPENALIZED-by-the-centered-metric mode.
    if !label_layout.joint_specs.is_empty() {
        let total_compiled: usize = specs.iter().map(|s| s.design.ncols()).sum();
        let joint_log_lambdas = label_layout.joint_log_lambdas(&rho_star);
        let bundle = gam_problem::JointPenaltyBundle::new(
            std::sync::Arc::new(label_layout.joint_specs.clone()),
            joint_log_lambdas,
            total_compiled,
        )
        .map_err(CustomFamilyError::from)?;
        final_options.joint_penalties = Some(std::sync::Arc::new(bundle));
    }
    let mut inner = inner_blockwise_fit(
        family,
        specs,
        &per_block,
        &final_options,
        final_seed.as_ref(),
    )
    .map_err(|error| CustomFamilyError::Optimization {
        context: "fit_custom_family final inner refit",
        reason: format!(
            "{error}; rho_checkpoint={:?}; no fit was assembled.{}",
            rho_star.as_slice().unwrap_or(&[]),
            last_error_detail
        ),
    })?;
    if !inner.converged {
        // Preserve the refit's rho/coefficients in the response-keyed cache
        // when this family supports persistent warm starts, then reject the
        // non-mode. The rho checkpoint is carried in the typed error regardless.
        store_persistent_custom_family_warm_start(
            persistent_warm_start_key.as_deref(),
            specs,
            &constrained_warm_start_from_inner(&rho_star, &inner),
        );
        return Err(CustomFamilyError::Optimization {
            context: "fit_custom_family final inner refit",
            reason: format!(
                "outer smoothing optimization final inner refit did not converge after {} cycles; \
                 rho_checkpoint={:?}; no fit was assembled.{}",
                inner.cycles,
                rho_star.as_slice().unwrap_or(&[]),
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
    audit_converged_identifiability(
        family,
        raw_specs,
        &canonical,
        &inner.block_states,
        outer_iters,
    )?;
    // gam#1587: pass `final_options` (carrying the joint penalty bundle) so the
    // posterior precision `H = H_lik + S_λ` includes the full-width centered
    // penalty, matching the inner-converged mode.
    let covariance_conditional = compute_joint_covariance_required(
        family,
        specs,
        &inner.block_states,
        &per_block,
        &final_options,
    )
    .map_err(|error| CustomFamilyError::Optimization {
        context: "fit_custom_family final covariance factorization",
        reason: format!(
            "{error}; rho_checkpoint={:?}; no fit was assembled",
            rho_star.as_slice().unwrap_or(&[])
        ),
    })?;

    // gam#1587/#561: pass `final_options` (carrying the joint penalty bundle at
    // the selected ρ*) so the exported geometry's penalized Hessian — and the
    // trace EDF derived from it — includes the full-width centered penalty,
    // matching the covariance path above and the inner-converged mode.
    let geometry = compute_joint_geometry(
        family,
        specs,
        &inner.block_states,
        &per_block,
        &final_options,
    )
    .map_err(|reason| CustomFamilyError::Optimization {
        context: "fit_custom_family joint geometry",
        reason,
    })?;
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
    // Cross-fit FitArtifact capture (Phase 0/1) for the converged smoothing
    // fit: persist the descriptor-indexed raw-β + ρ so a later fold transfers
    // ρ. Best-effort; never affects this fit's result. Gated on the same opt-in
    // as the consume side (gam#1607) so opt-out families publish nothing and
    // stay bit-reproducible across repeat fits/runs.
    if cross_fit_artifact_enabled {
        capture_fit_artifact::<F>(
            specs,
            &canonical.gauge,
            &final_warm_start.block_beta,
            &final_warm_start.rho,
            &label_layout.physical_to_outer,
            penalized_objective,
            true,
        );
    }
    let rho_star_physical = expand_labeled_log_lambdas(&rho_star, &label_layout)?;
    // gam#1587/#561: a family whose smoothing rides on the full-width JOINT
    // penalty (the multinomial centered `Σ_t λ_t (M ⊗ S_t)` metric) leaves its
    // per-block penalty lists — and hence the physical `rho_physical`/`lambdas`
    // expansion above — EMPTY, so the selected per-component `ρ_t` would be lost
    // at assembly. Surface it on `FitArtifacts.joint_log_lambdas` so the
    // reporting path can rebuild per-(class, term) λ and the influence-matrix
    // EDF. `None` (no allocation) for every per-block-only family.
    let joint_log_lambdas = (!label_layout.joint_specs.is_empty())
        .then(|| Array1::from(label_layout.joint_log_lambdas(&rho_star)));
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
            outer_converged: true,
            joint_log_lambdas,
        },
    )
}

pub fn fit_custom_family_fixed_log_lambdas<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    raw_specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
    warm_start: Option<&CustomFamilyWarmStart>,
    outer_iterations: usize,
    outer_gradient_norm: Option<f64>,
    outer_converged: bool,
) -> Result<gam_solve::model_types::UnifiedFitResult, CustomFamilyError> {
    if !outer_converged {
        return Err(CustomFamilyError::Optimization {
            context: "fit_custom_family_fixed_log_lambdas",
            reason: "the enclosing outer optimization did not certify convergence; refusing \
                     to run inference or assemble a fixed-lambda fit from its checkpoint"
                .to_string(),
        });
    }
    let canonical = gam_identifiability::canonical::canonicalize_for_identifiability(raw_specs)?;
    let specs: &[ParameterBlockSpec] = &canonical.reduced_specs;
    let penalty_counts = validate_blockspecs(specs)?;
    let rho = flatten_log_lambdas(specs);
    let per_block = split_log_lambdas(&rho, &penalty_counts)?;
    let reduced_warm_start = fixed_lambda_warm_start_for_reduced_specs(warm_start, &canonical);
    let mut inner = inner_blockwise_fit(family, specs, &per_block, options, reduced_warm_start)
        .map_err(|error| CustomFamilyError::Optimization {
            context: "fit_custom_family_fixed_log_lambdas inner solve",
            reason: format!(
                "{error}; rho_checkpoint={:?}; no fit was assembled",
                rho.as_slice().unwrap_or(&[])
            ),
        })?;
    if !inner.converged {
        return Err(CustomFamilyError::Optimization {
            context: "fit_custom_family_fixed_log_lambdas inner solve",
            reason: format!(
                "fixed-log-lambda inner solve did not converge after {} cycles; \
                 rho_checkpoint={:?}; no fit was assembled",
                inner.cycles,
                rho.as_slice().unwrap_or(&[])
            ),
        });
    }
    refresh_all_block_etas(family, specs, &mut inner.block_states)?;
    audit_converged_identifiability(
        family,
        raw_specs,
        &canonical,
        &inner.block_states,
        outer_iterations,
    )?;
    let covariance_conditional =
        compute_joint_covariance_required(family, specs, &inner.block_states, &per_block, options)
            .map_err(|error| CustomFamilyError::Optimization {
                context: "fit_custom_family_fixed_log_lambdas covariance factorization",
                reason: format!(
                    "{error}; rho_checkpoint={:?}; no fit was assembled",
                    rho.as_slice().unwrap_or(&[])
                ),
            })?;
    let geometry = compute_joint_geometry(family, specs, &inner.block_states, &per_block, options)
        .map_err(|reason| CustomFamilyError::Optimization {
            context: "fit_custom_family_fixed_log_lambdas joint geometry",
            reason,
        })?;
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
            canonical: Some(&canonical),
            result_specs: raw_specs,
            penalized_objective,
            outer_iterations,
            outer_gradient_norm,
            criterion_certificate: None,
            outer_converged: true,
            joint_log_lambdas: None,
        },
    )
}

pub fn fit_custom_family_fixed_log_lambda_warm_start<
    F: CustomFamily + Clone + Send + Sync + 'static,
>(
    family: &F,
    raw_specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
) -> Result<(Vec<Array1<f64>>, bool, usize), CustomFamilyError> {
    let canonical = gam_identifiability::canonical::canonicalize_for_identifiability(raw_specs)?;
    let specs: &[ParameterBlockSpec] = &canonical.reduced_specs;
    let penalty_counts = validate_blockspecs(specs)?;
    let rho = flatten_log_lambdas(specs);
    let per_block = split_log_lambdas(&rho, &penalty_counts)?;
    let inner = inner_blockwise_fit(family, specs, &per_block, options, None)?;
    let theta_blocks: Vec<Array1<f64>> = inner
        .block_states
        .iter()
        .map(|state| state.beta.clone())
        .collect();
    let block_beta = canonical.gauge.lift_block_betas(&theta_blocks);
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
