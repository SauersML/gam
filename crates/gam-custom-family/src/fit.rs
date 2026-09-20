//! The public fit entry points (`fit_custom_family`,
//! `fit_custom_family_with_rho_prior`, fixed-lambda variants), result assembly +
//! output-channel wiring, the raw-coordinate lift, and the effective-df-floor
//! rho-bound machinery.

use super::*;

/// A descending ray's direction lifted from the inner solve's reduced coordinates
/// to raw joint order through the identifiability gauge: `d = T·δ`, with no affine
/// shift, because a direction is a difference of coefficients (#979).
pub(crate) fn lift_direction_to_raw(
    gauge: &gam_problem::gauge::Gauge,
    reduced: &[f64],
) -> std::sync::Arc<[f64]> {
    gauge
        .t_full
        .dot(&ndarray::ArrayView1::from(reduced))
        .iter()
        .copied()
        .collect()
}

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

/// Operating-point `family_scalars` for the PRE-fit identifiability audit, built
/// from each block spec's warm-start `initial_beta` (the pilot β the family
/// seeded; zeros where it seeded none).
///
/// A family whose effective channel weights depend on β — survival marginal-slope,
/// `c_i = √(1+(s·g_i)²)` — collapses to its raw design when linearized at β = 0,
/// aliasing structurally-identical blocks that are only distinguished by that
/// weighting and producing a FALSE identifiability refusal before the fit even
/// starts. Linearizing the pre-fit audit at the pilot operating point (the same
/// geometry `audit_converged_identifiability` uses post-convergence) ranks the
/// design the fit actually sees. Static families return `None` (the trait
/// default) and the audit linearizes at the zero/init point exactly as before.
///
/// `eta` is unused by `current_identifiability_family_scalars` (it reads β only),
/// so a zero placeholder keeps each synthetic state well-formed.
fn pre_fit_operating_scalars<F: CustomFamily + ?Sized>(
    family: &F,
    specs: &[ParameterBlockSpec],
) -> Result<Option<Arc<dyn std::any::Any + Send + Sync>>, CustomFamilyError> {
    let states: Vec<ParameterBlockState> = specs
        .iter()
        .map(|spec| {
            let beta = spec
                .initial_beta
                .clone()
                .unwrap_or_else(|| Array1::zeros(spec.design.ncols()));
            let eta = Array1::zeros(spec.design.nrows());
            ParameterBlockState { beta, eta }
        })
        .collect();
    family
        .current_identifiability_family_scalars(&states)
        .map_err(|reason| CustomFamilyError::Optimization {
            context: "pre-fit identifiability operating scalars",
            reason,
        })
}

/// Ask the family, block by block, whether that block's COEFFICIENT COORDINATE
/// is model content or only its column space is (#2748).
///
/// The identifiability canonicaliser decides whether it may reparameterise a
/// block — `β ↦ Vᵀβ`, penalties pulled back as `VᵀSV` — and that is exact only
/// when any basis of the block's column space is the same model. It is not for a
/// monotone link-wiggle warp, whose family imposes `β_w ≥ 0` componentwise on
/// those very coefficients and rebuilds its design at that exact width. The
/// canonicaliser takes only specs, so the family has to be asked here, at the
/// one site that holds both.
///
/// The probe state mirrors [`pre_fit_operating_scalars`]: the warm start when
/// there is one, zeros otherwise, at RAW block width — which is what every
/// implementor of `block_linear_constraints` validates its index and width
/// against, so the derived default in
/// [`CustomFamily::block_coefficient_coordinate`] can answer for every family
/// instead of panicking on an empty state slice.
fn pre_fit_coefficient_coordinates<F: CustomFamily + ?Sized>(
    family: &F,
    specs: &[ParameterBlockSpec],
) -> Vec<CoefficientCoordinate> {
    let states: Vec<ParameterBlockState> = specs
        .iter()
        .map(|spec| {
            let beta = spec
                .initial_beta
                .clone()
                .unwrap_or_else(|| Array1::zeros(spec.design.ncols()));
            let eta = Array1::zeros(spec.design.nrows());
            ParameterBlockState { beta, eta }
        })
        .collect();
    let coordinates: Vec<CoefficientCoordinate> = specs
        .iter()
        .enumerate()
        .map(|(index, spec)| family.block_coefficient_coordinate(&states, index, spec))
        .collect();
    let structural: Vec<&str> = specs
        .iter()
        .zip(coordinates.iter())
        .filter(|(_, coordinate)| coordinate.is_structural())
        .map(|(spec, _)| spec.name.as_str())
        .collect();
    if !structural.is_empty() {
        log::debug!(
            "[CANON] structural coefficient coordinate(s) declared: [{}] — these blocks keep \
             their basis AND their width through canonicalisation (#2748)",
            structural.join(", "),
        );
    }
    coordinates
}

/// Does the family declare a linear inequality on any block (gam#2765)?
///
/// Such a family's criterion prices the constrained Laplace normalizer, which the EFS fixed point
/// does not model, so its outer search takes the gradient route. The probe state is
/// [`pre_fit_coefficient_coordinates`]'s. A family that cannot answer there is treated as
/// constrained: the gradient route is valid for every family, the fixed point is not.
fn pre_fit_declares_linear_constraints<F: CustomFamily + ?Sized>(
    family: &F,
    specs: &[ParameterBlockSpec],
) -> bool {
    let states: Vec<ParameterBlockState> = specs
        .iter()
        .map(|spec| {
            let beta = spec
                .initial_beta
                .clone()
                .unwrap_or_else(|| Array1::zeros(spec.design.ncols()));
            let eta = Array1::zeros(spec.design.nrows());
            ParameterBlockState { beta, eta }
        })
        .collect();
    specs.iter().enumerate().any(|(index, spec)| {
        !matches!(family.block_linear_constraints(&states, index, spec), Ok(None))
    })
}

/// The `(pilot, current)` β pair the converged drift audit prices against each
/// other, flattened over blocks in spec order.
///
/// The pilot is the operating point the PRE-FIT audit linearized at, and
/// [`pre_fit_operating_scalars`] builds that from `spec.initial_beta` — zeros
/// are its fallback for a block with no warm start, not the pilot itself.
///
/// Handing the drift audit a bare zero vector instead is not a cosmetic slip.
/// `maybe_log_audit_drift` publishes
/// `beta_relative_change = ‖β̂ − β₀‖ / (‖β₀‖ + f64::EPSILON)`, so a zeros
/// reference puts MACHINE EPSILON in the denominator and the reported number is
/// ~1e16 for every warm-started fit regardless of how far it actually
/// travelled — it cannot distinguish the two cases it exists to distinguish
/// (#2360).
///
/// Both halves live here so there is one place that decides the pair, and so
/// the contract is reachable from a test: `audit_converged_identifiability` has
/// no other route to it.
pub(crate) fn drift_audit_beta_pair(
    specs: &[ParameterBlockSpec],
    raw_states: &[ParameterBlockState],
) -> (Vec<f64>, Vec<f64>) {
    let pilot: Vec<f64> = specs
        .iter()
        .flat_map(|spec| match spec.initial_beta.as_ref() {
            Some(beta) => beta.to_vec(),
            None => vec![0.0; spec.design.ncols()],
        })
        .collect();
    let current: Vec<f64> = raw_states
        .iter()
        .flat_map(|state| state.beta.iter().copied())
        .collect();
    (pilot, current)
}

/// Re-run the identifiability audit at the converged raw-coordinate state when a
/// family exposes dynamic primary scalars, and refuse on any verdict change.
///
/// The converged verdict is measured by the SAME audit function that produced the
/// pilot verdict. For a channel-aware family that is
/// `channel_aware_audit_at_operating_scalars`, re-run at the converged operating
/// scalars over the raw specs and over the canonical reduced specs. Re-measuring with
/// the penalty-augmented flat audit instead reported every penalty-covered design
/// alias as recovered, so an unchanged model refused after convergence (#2627 finding
/// 9). Flat-routed families keep the flat audit.
///
/// A channel-aware verdict refuses when a rank or fatality changes: the raw problem's,
/// or that of the problem the fit ran through the pilot's gauge. A drop set that names
/// other members of the same alias classes is a representative swap. It is logged, and
/// it does not refuse. See `ConvergedChannelAwareVerdict::refuses`.
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
    let (beta_pilot, beta_current) = drift_audit_beta_pair(raw_specs, &raw_states);
    let (drift, refuses, pilot_gauge_rank) = if canonical.used_channel_aware_audit {
        let verdict = gam_identifiability::canonical::converged_channel_aware_verdict(
            raw_specs,
            canonical,
            Arc::clone(&family_scalars),
            gam_identifiability::audit::audit_beta_relative_change(&beta_pilot, &beta_current),
            outer_iter,
        )?;
        let refuses = verdict.refuses();
        let pilot_gauge_rank = verdict.pilot_gauge_rank();
        if verdict.representative_swap() {
            let newly_dropped: Vec<String> = verdict
                .drift
                .newly_dropped
                .iter()
                .map(|dropped| format!("{}[{}]", dropped.block, dropped.column))
                .collect();
            log::debug!(
                "[AUDIT-DRIFT] converged identifiability accepted a representative swap: the pivot \
                 chose other members of the same alias classes (rank {}, pilot gauge rank {} at \
                 convergence); newly_dropped=[{}] recovered=[{}]",
                verdict.drift.current_rank,
                pilot_gauge_rank,
                newly_dropped.join(", "),
                verdict.drift.recovered.join(", "),
            );
        }
        (verdict.drift, refuses, Some(pilot_gauge_rank))
    } else {
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
        let refuses = drift.verdict_changed();
        (drift, refuses, None)
    };
    if refuses {
        return Err(CustomFamilyError::Optimization {
            context: "converged identifiability audit",
            reason: format!(
                "identifiability verdict changed after convergence: pilot_rank={} current_rank={} pilot_fatal={} current_fatal={} newly_dropped={} recovered={} pilot_gauge_rank={}",
                drift.pilot_rank,
                drift.current_rank,
                drift.pilot_fatal,
                drift.current_fatal,
                drift.newly_dropped.len(),
                drift.recovered.len(),
                pilot_gauge_rank.map_or_else(|| "n/a (flat audit)".to_string(), |rank| rank.to_string()),
            ),
        });
    }
    // #2360 (#2337 §8 Thm 8.3) — say WHY this accepted.
    //
    // Agreeing endpoint ranks is a statement about two points, and the fit
    // lives on the path between them: the pilot verdict can hold at both ends
    // while the interior went somewhere else entirely. When the pilot's
    // certificate transported, the agreement is a genuine path guarantee. When
    // its margin was provably exhausted, the agreement is a coincidence that
    // happens to be the right answer, and the record should not read as more
    // than that. The verdict stays as-is either way — this is the honest
    // provenance of an acceptance, not a new refusal.
    match drift.pilot_certificate_transported {
        Some(false) => {
            let (excursion, radius) = drift.excursion_vs_radius.unwrap_or((f64::NAN, f64::NAN));
            log::debug!(
                "[AUDIT-TRANSPORT] converged identifiability accepted on ENDPOINT AGREEMENT \
                 ONLY (rank={}): the pilot certificate's transport radius {radius:.3e} was \
                 exhausted by an excursion of at least {excursion:.3e}, so the pilot verdict is \
                 not certified along the path it travelled",
                drift.current_rank
            );
        }
        Some(true) => {
            let (excursion, radius) = drift.excursion_vs_radius.unwrap_or((f64::NAN, f64::NAN));
            log::trace!(
                "[AUDIT-TRANSPORT] converged identifiability rank={} TRANSPORTED from the pilot \
                 (excursion {excursion:.3e} within radius {radius:.3e})",
                drift.current_rank
            );
        }
        None => {}
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
    /// `CustomFamily::classical_deviance` at the converged mode; `None` when
    /// the family declares none (see `BlockwiseFitResultParts::deviance`).
    pub(crate) deviance: Option<f64>,
    pub(crate) covariance_conditional: Option<Array2<f64>>,
    pub(crate) geometry: Option<FitGeometry>,
    /// EDF derived in the reduced coefficient frame before a non-square gauge
    /// lift. Row-wise working evidence is orthogonal to this precision path.
    pub(crate) precomputed_edf:
        Option<(f64, Vec<f64>, Vec<f64>, Vec<f64>, Vec<gam_solve::estimate::EdfRankBound>)>,
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
    /// First-order ρ-uncertainty smoothing correction `C` (in the REDUCED
    /// coefficient frame; lifted through the gauge alongside the conditional
    /// covariance) with its typed provenance (#2346). `V_c = V_cond + C` is
    /// published as `beta_covariance_corrected`. `None` when no outer ρ
    /// curvature was retained or the interior V_ρ is not honestly finite.
    pub(crate) smoothing_corrected: Option<(
        Array2<f64>,
        gam_solve::model_types::SmoothingCorrectionMethod,
    )>,
    /// Why no correction was minted on a fit that selected ρ (#2677).
    pub(crate) smoothing_correction_absence: Option<gam_solve::model_types::SmoothingCorrectionAbsence>,
    /// Which rule selected the coefficient mode the fit reports (#2366, #2661).
    pub(crate) coefficient_mode_selection: gam_solve::model_types::CoefficientModeSelection,
    /// The terminal posterior assembly's improper-posterior evidence (#3164),
    /// published on `FitArtifacts::improper_penalty_null_posterior`.
    pub(crate) improper_penalty_null_posterior:
        Option<gam_problem::jeffreys_arming::JeffreysArmingEvidence>,
}

/// The family's classical deviance at the converged mode, as a typed
/// `CustomFamilyError` so the four assembly sites share one mapping.
fn classical_deviance_at_mode<F: CustomFamily + ?Sized>(
    family: &F,
    block_states: &[ParameterBlockState],
    context: &'static str,
) -> Result<Option<f64>, CustomFamilyError> {
    family
        .classical_deviance(block_states)
        .map_err(|reason| CustomFamilyError::Optimization { context, reason })
}

pub(crate) fn assemble_custom_family_fit_result(
    inner: BlockwiseInnerResult,
    assembly: BlockwiseFitAssembly<'_>,
) -> Result<gam_solve::model_types::UnifiedFitResult, CustomFamilyError> {
    let BlockwiseFitAssembly {
        rho_physical,
        deviance,
        covariance_conditional,
        geometry,
        precomputed_edf,
        canonical,
        result_specs,
        penalized_objective,
        outer_iterations,
        outer_gradient_norm,
        criterion_certificate,
        outer_converged,
        joint_log_lambdas,
        smoothing_corrected,
        smoothing_correction_absence,
        coefficient_mode_selection,
        improper_penalty_null_posterior,
    } = assembly;
    let log_lambdas = rho_physical;
    let lambdas =
        exact_lambdas_from_log_strengths(&log_lambdas, "custom-family fitted log strength")?;
    let (block_states, covariance_conditional, geometry, precomputed_edf, smoothing_corrected) =
        if let Some(canonical) = canonical {
            let precomputed_edf = precomputed_edf
                .or_else(|| reduced_blockwise_edf(geometry.as_ref(), canonical, &lambdas));
            let block_states = lift_block_states_to_raw(canonical, inner.block_states);
            let (covariance_conditional, geometry) =
                lift_fit_geometry_to_raw(canonical, covariance_conditional, geometry)?;
            // The correction is a coefficient-space bilinear form exactly like
            // the conditional covariance: same gauge congruence (#2346).
            let smoothing_corrected = smoothing_corrected
                .map(|(c, method)| (canonical.gauge.lift_covariance(&c), method));
            (
                block_states,
                covariance_conditional,
                geometry,
                precomputed_edf,
                smoothing_corrected,
            )
        } else {
            (
                inner.block_states,
                covariance_conditional,
                geometry,
                precomputed_edf,
                smoothing_corrected,
            )
        };

    let mut fit = blockwise_fit_from_parts(
        BlockwiseFitResultParts {
            block_states,
            log_likelihood: inner.log_likelihood,
            deviance,
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
            smoothing_corrected,
            smoothing_correction_absence,
        },
        result_specs,
    )?;
    fit.artifacts.coefficient_mode_selection = coefficient_mode_selection;
    fit.artifacts.improper_penalty_null_posterior = improper_penalty_null_posterior;
    Ok(fit)
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
            design: Arc::new(dense),
            own_output,
            n_family_outputs,
        }));
    }
    Ok(Some(wired))
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
    penalty_range_gammas_from_gram(&gram, &s_dense)
}

pub use gam_solve::estimate::rho_domain::{penalty_range_gammas_from_gram, resolvability_interval};

/// The ρ at which one penalized term is switched off to the gradient's resolution —
/// the upper edge of its resolvability interval — for a caller that seeds a
/// term at its null space (the constant-scale AFT time warp). `None` when the
/// term carries no data curvature against its penalty.
pub fn penalized_term_switch_off_rho(design: &DesignMatrix, penalty: &Array2<f64>) -> Option<f64> {
    let p = design.ncols();
    if p == 0 || penalty.nrows() != p || penalty.ncols() != p {
        return None;
    }
    let x = design.to_dense();
    let gram = x.t().dot(&x);
    let gammas = penalty_range_gammas_from_gram(&gram, penalty)?;
    resolvability_interval(&gammas).map(|(_, upper)| upper.min(gam_problem::LOG_STRENGTH_MAX))
}

fn include_interval(slot: &mut (f64, f64), seen: &mut bool, interval: (f64, f64)) {
    if *seen {
        slot.0 = slot.0.min(interval.0);
        slot.1 = slot.1.max(interval.1);
    } else {
        *slot = interval;
        *seen = true;
    }
}

/// Per outer coordinate, the ρ interval on which the terms it drives are
/// resolvable against their own design curvature — the λ-selection domain
/// (#2812). This replaces the hand box (`[rho_lower_bound, 12]` tightened to an
/// effective-df floor of one) whose walls the search used to hit: a search that
/// reaches an edge of THIS domain has found a term that is unpenalized, or one
/// that has collapsed to its penalty null space (`edf → nullspace_dim`), to
/// the gradient's resolution. That is a result, not a wall, and the certificate reads
/// it as one.
///
/// Each physical penalty contributes its own [`resolvability_interval`]. A
/// shared coordinate must remain free while ANY of its terms is resolvable,
/// so its domain spans their union. Joint penalties (one carrier
/// shared across blocks, #2579) are read off the stacked block Gram, and a
/// group of joint penalties sharing an outer coordinate spans the union
/// of its members' intervals. A coordinate no projectable geometry reaches
/// keeps the precision box `[ln ε, ln(1/ε)]` around unit strength. Everything
/// is intersected with the representable log-strength range, and a
/// family-declared floor (`options.rho_lower_bound`: the multinomial's derived
/// minimum strength) raises the lower edge; the engine adds no number of its
/// own.
pub(crate) fn resolvability_rho_domain(
    specs: &[ParameterBlockSpec],
    layout: &PenaltyLabelLayout,
    n_rho: usize,
    family_floor: Option<f64>,
) -> Result<(Array1<f64>, Array1<f64>), CustomFamilyError> {
    resolvability_rho_domain_and_limit_faces(specs, layout, n_rho, family_floor)
        .map(|domain| (domain.lower, domain.upper))
}

/// [`resolvability_rho_domain`] with the kind of each face (#2954), by the rule
/// `gam_solve::estimate::rho_domain` applies to a standard REML design: an edge
/// is its terms' limit model when it is their own resolvability edge. Past the
/// upper edge every term on the coordinate sits at its penalty null-space fit;
/// below the lower edge every term sits at its unpenalized fit, which is a limit
/// only where each term's data identify it
/// (`rho_domain::unpenalized_fit_is_identified`). The precision box of a
/// coordinate no geometry reaches, an edge the representable log-strength range
/// cut, and a family floor are literals, not limits.
pub(crate) struct ResolvabilityRhoDomain {
    pub(crate) lower: Array1<f64>,
    pub(crate) upper: Array1<f64>,
    pub(crate) lower_is_limit: Vec<bool>,
    pub(crate) upper_is_limit: Vec<bool>,
}

pub(crate) fn resolvability_rho_domain_and_limit_faces(
    specs: &[ParameterBlockSpec],
    layout: &PenaltyLabelLayout,
    n_rho: usize,
    family_floor: Option<f64>,
) -> Result<ResolvabilityRhoDomain, CustomFamilyError> {
    use gam_solve::estimate::rho_domain::unpenalized_fit_is_identified;
    let precision_box = gam_solve::estimate::rho_domain::precision_box();
    let mut intervals: Vec<(f64, f64)> = vec![precision_box; n_rho];
    let mut seen = vec![false; n_rho];
    let mut identified = vec![true; n_rho];
    let mut physical = 0usize;
    for spec in specs {
        if spec.penalties.is_empty() {
            continue;
        }
        let p = spec.design.ncols();
        let mut aggregate = Array2::<f64>::zeros((p, p));
        for penalty in &spec.penalties {
            let dense = penalty.to_dense();
            if dense.dim() == aggregate.dim() {
                aggregate += &dense;
            }
        }
        let x = spec.design.to_dense();
        let gram = x.t().dot(&x);
        for penalty in &spec.penalties {
            let outer = layout.physical_to_outer.get(physical).copied().flatten();
            physical += 1;
            let Some(outer) = outer else {
                continue; // fixed penalty: not an outer coordinate.
            };
            if outer >= n_rho {
                continue;
            }
            let gammas = if spec.penalties.len() == 1 {
                design_penalty_range_gammas(&spec.design, penalty)
            } else {
                gam_solve::estimate::rho_domain::penalty_range_gammas_with_shared_nullspace(
                    &gram, &penalty.to_dense(), &aggregate,
                )
            };
            let Some((gammas, interval)) = gammas.and_then(|gammas| {
                resolvability_interval(&gammas).map(|interval| (gammas, interval))
            }) else {
                continue; // un-projectable geometry: the precision box stands.
            };
            include_interval(&mut intervals[outer], &mut seen[outer], interval);
            identified[outer] &= unpenalized_fit_is_identified(&gammas, p);
        }
    }
    if !layout.joint_specs.is_empty()
        && let Some(joint_gram) = stacked_block_design_gram(specs)
    {
        let mut aggregate = Array2::<f64>::zeros(joint_gram.dim());
        for joint in layout.joint_specs.iter() {
            aggregate += &joint.matrix;
        }
        let mut offset = 0;
        for spec in specs {
            let p = spec.design.ncols();
            for penalty in &spec.penalties {
                let dense = penalty.to_dense();
                if dense.dim() == (p, p) {
                    let mut block = aggregate.slice_mut(ndarray::s![offset..offset+p, offset..offset+p]);
                    block += &dense;
                }
            }
            offset += p;
        }
        let columns = joint_gram.nrows();
        let mut group_intervals: Vec<(usize, (f64, f64), bool)> = Vec::new();
        for (joint_idx, joint) in layout.joint_specs.iter().enumerate() {
            let Some((gammas, interval)) =
                gam_solve::estimate::rho_domain::penalty_range_gammas_with_shared_nullspace(
                    &joint_gram,
                    &joint.matrix,
                    &aggregate,
                )
                .and_then(|gammas| {
                    resolvability_interval(&gammas).map(|interval| (gammas, interval))
                })
            else {
                continue;
            };
            let joint_identified = unpenalized_fit_is_identified(&gammas, columns);
            match joint.group {
                Some(group) => match group_intervals.iter_mut().find(|(g, _, _)| *g == group) {
                    Some((_, current, group_identified)) => {
                        current.0 = current.0.min(interval.0);
                        current.1 = current.1.max(interval.1);
                        *group_identified &= joint_identified;
                    }
                    None => group_intervals.push((group, interval, joint_identified)),
                },
                None => {
                    if let Some(&outer) = layout.joint_to_outer.get(joint_idx)
                        && outer < n_rho
                    {
                        include_interval(&mut intervals[outer], &mut seen[outer], interval);
                        identified[outer] &= joint_identified;
                    }
                }
            }
        }
        for (joint_idx, joint) in layout.joint_specs.iter().enumerate() {
            let Some(group) = joint.group else { continue };
            let Some(&(_, interval, group_identified)) =
                group_intervals.iter().find(|(g, _, _)| *g == group)
            else {
                continue;
            };
            let Some(&outer) = layout.joint_to_outer.get(joint_idx) else {
                continue;
            };
            if outer < n_rho {
                include_interval(&mut intervals[outer], &mut seen[outer], interval);
                identified[outer] &= group_identified;
            }
        }
    }
    let mut lower = Array1::<f64>::zeros(n_rho);
    let mut upper = Array1::<f64>::zeros(n_rho);
    let mut lower_is_limit = vec![false; n_rho];
    let mut upper_is_limit = vec![false; n_rho];
    for (outer, interval) in intervals.iter().enumerate() {
        let mut lo = interval.0.max(gam_problem::LOG_STRENGTH_MIN);
        let hi = interval.1.min(gam_problem::LOG_STRENGTH_MAX);
        if let Some(floor) = family_floor {
            lo = lo.max(floor);
        }
        if !(lo.is_finite() && hi.is_finite() && lo <= hi) {
            return Err(CustomFamilyError::InvalidInput {
                context: "resolvability rho domain",
                reason: format!(
                    "coordinate {outer}: the family floor {family_floor:?} lies above the \
                     term's resolvability ceiling {hi} (interval {interval:?}); the \
                     admissible domain is empty"
                ),
            });
        }
        lower[outer] = lo;
        upper[outer] = hi;
        lower_is_limit[outer] = seen[outer] && identified[outer] && lo == interval.0;
        upper_is_limit[outer] = seen[outer] && hi == interval.1;
    }
    log::trace!(
        "[RHO-DOMAIN] resolvability domain per coordinate: lower={:.3?} upper={:.3?} \
         lower_is_limit={lower_is_limit:?} upper_is_limit={upper_is_limit:?}",
        lower.to_vec(),
        upper.to_vec(),
    );
    Ok(ResolvabilityRhoDomain {
        lower,
        upper,
        lower_is_limit,
        upper_is_limit,
    })
}

/// The #2812 ρ domain of per-block penalties, by the same law
/// [`fit_custom_family`] applies to its outer search, for a driver that
/// searches these blocks' ρ jointly with other hyperparameters and would
/// otherwise keep the precision box (the spatial length-scale route, #2896).
/// Joint cross-block penalties are not part of `specs`, so none are folded in.
pub fn per_block_resolvability_rho_domain(
    specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
) -> Result<(Array1<f64>, Array1<f64>), CustomFamilyError> {
    let penalty_counts = validate_blockspecs(specs)?;
    let label_layout = penalty_label_layout_with_joint(specs, penalty_counts, Vec::new())?;
    let n_rho = label_layout.initial_rho.len();
    resolvability_rho_domain(specs, &label_layout, n_rho, options.rho_lower_bound)
}

/// Unit-weight Gram of the STACKED block parameter vector, `blkdiag(X_bᵀ X_b)`.
///
/// A joint penalty matrix is `(total_compiled, total_compiled)`: it acts on the
/// concatenation of every block's coefficients, in block order. The design that
/// matches that vector is the block-diagonal stacking of the per-block designs,
/// so its Gram is the block diagonal of the per-block Grams — the off-diagonal
/// blocks are structurally zero, because block `b`'s design columns multiply
/// only block `b`'s coefficients.
///
/// Returns `None` when a block's design cannot be densified to its declared
/// width, so the caller keeps the uniform bound rather than bounding ρ against a
/// mis-shaped curvature.
fn stacked_block_design_gram(specs: &[ParameterBlockSpec]) -> Option<Array2<f64>> {
    let total: usize = specs.iter().map(|spec| spec.design.ncols()).sum();
    if total == 0 {
        return None;
    }
    let mut gram = Array2::<f64>::zeros((total, total));
    let mut offset = 0usize;
    for spec in specs {
        let p = spec.design.ncols();
        if p == 0 {
            continue;
        }
        let x = spec.design.to_dense();
        if x.ncols() != p {
            return None;
        }
        let block = x.t().dot(&x);
        for i in 0..p {
            for j in 0..p {
                gram[(offset + i, offset + j)] = block[(i, j)];
            }
        }
        offset += p;
    }
    Some(gram)
}

/// Seed the outer search with the mode the *definition* of `θ̂(ρ)` names, rather
/// than with whatever mode the inner solver happens to reach from the caller's
/// coefficients (#2366).
///
/// # Why a definition is needed at all
///
/// The outer criterion is profiled: `V(ρ) = ℓ_p(θ̂(ρ), ρ)` with
/// `θ̂(ρ) ∈ argmin_θ ℓ_p(θ, ρ)`. `V` is a *function of ρ* only when that `argmin`
/// is single-valued, or when a selection rule fixes which element is meant. For
/// for a family whose complete coefficient objective is nonconvex — link-wiggle
/// (the warp basis is evaluated at the current index), transformation models
/// with an `I`-spline cone, constrained PIRLS, locscale with a nonlinear warp —
/// `argmin ℓ_p(·, ρ)` can be a set of disconnected modes. Without a selection rule,
/// `θ̂` is a functional of the solver's trajectory and of the persistent cache,
/// so `V` is not a function, and three things follow that this codebase has been
/// treating as separate defects:
///
/// * the same data and model fit differently depending on cache state (#2363);
/// * the envelope identity `dV/dρ = ∂ℓ_p/∂ρ|_θ̂` holds only along a branch that
///   is continuous in ρ, so evaluations that land on different branches produce
///   a gradient describing one branch and a value sequence walking another —
///   the objective↔gradient desync class (#2349, #1561, #2298);
/// * a gradient method cannot certify stationarity of an object that changes
///   under it, so `‖Pg‖` stalls at `O(branch gap / step)` instead of `→ 0`.
///
/// # The selection rule
///
/// `θ̂(ρ)` is defined as the endpoint of the continuation that starts at the
/// anchor `ρ_A` and follows the segment `ρ(t) = ρ_A + t·(ρ − ρ_A)`, `t: 0 → 1`.
/// The anchor is the MAXIMAL-smoothing ρ, the per-coordinate ceiling of the
/// derived [`resolvability_rho_domain`]: there every penalized term is
/// switched off to the gradient's resolution, collapsed onto its penalty nullspace,
/// the surviving low-dimensional problem is the parametric fit, and its mode
/// is unique — which is the one property the selection rule needs, since a
/// unique mode is what the caller's coefficients cannot influence.
///
/// Before #2812 the anchor was a hand ceiling tightened to an effective-df
/// floor, which answered a different question — how much df a term must
/// RETAIN — and #2608 made it a partial collapse for rank-deficient terms. A
/// partially collapsed term is still
/// nonconvex, so anchoring there would return branch selection to the seed.
///
/// The mathematical object is the limit of the exact continuation path, so the
/// discretization is refined — 1, 2, 4, … uniform steps — until the endpoint
/// stops moving. What "stops moving" means, and what it is measured against, is
/// derived in [`certify_refined_continuation`]: the endpoint sequence is
/// MODE-VALUED, not smoothly convergent, and both the yardstick and the stopping
/// rule follow from that.
///
/// # Typed termination
///
/// A corrector that does not converge, a ladder that exhausts its refinement
/// budget without the endpoint settling, or a path that reaches floating-point
/// resolution means the continuation does not name a mode. Those outcomes are
/// typed [`AnchoredContinuationRefusal`]s rather than an undifferentiated
/// `None`. Success carries an [`AnchoredContinuationCertificate`] that records
/// the refinement and the agreement which proved the endpoint invariant. The
/// production caller logs a refusal and keeps its existing seed, so declining a
/// continuation still never turns a fit that works today into a failure.

/// How many CONSECUTIVE refinements must name the same mode before the ladder
/// calls it the limit.
///
/// One agreement is provably not enough, and the counterexample is measured
/// rather than imagined. On the #2612 penguins stride-4 armed refit the ladder's
/// endpoint-criterion trail is
///
/// ```text
///   steps  2 -> 4    agree to 9.0e-7   <- and BOTH are on a mode the next
///   steps  4 -> 8    differ by 6.4e-2     refinement leaves
///   steps  8 -> 16   agree to 5.0e-7
/// ```
///
/// so a ladder that certified on the first agreement would have stopped at 4
/// steps and returned a mode the exact path does not reach. A dyadic sweep at
/// `2k` steps visits every waypoint of the `k`-step sweep and inserts one
/// midpoint between each pair, so an agreement says "inserting midpoints once
/// did not change the branch" and two consecutive agreements say it at two
/// levels.
///
/// There is no finite certificate that a mode-valued sequence has reached its
/// limit — this is an evidence standard, not a theorem, and it is set one level
/// above the measured failure. The count reached is recorded in the certificate
/// so that a future fixture which defeats it changes one number with its
/// evidence attached rather than a rule with none.
const REQUIRED_CONSECUTIVE_AGREEMENTS: usize = 2;

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct AnchoredContinuationCertificate {
    pub(crate) steps: usize,
    /// The state-space discrepancy (linear predictors, log-likelihood, penalty)
    /// at the certifying refinement. Reported as evidence; it is NOT the
    /// certifying quantity — see [`certify_refined_continuation`].
    pub(crate) endpoint_discrepancy: f64,
    pub(crate) inner_tolerance: f64,
    pub(crate) observed_contraction_factor: Option<f64>,
    /// Relative agreement of the criterion the seed exists to make well defined,
    /// between this refinement's endpoint and the previous one's. This is what
    /// the certificate is taken on.
    pub(crate) criterion_agreement: f64,
    /// The resolution `criterion_agreement` was required to clear: the outer
    /// solver's own relative-cost tolerance, in the criterion's own units.
    pub(crate) criterion_resolution: f64,
    /// How many consecutive refinements agreed, at least
    /// [`REQUIRED_CONSECUTIVE_AGREEMENTS`].
    pub(crate) consecutive_agreements: usize,
}

pub(crate) struct CertifiedAnchoredContinuationSeed {
    pub(crate) warm_start: ConstrainedWarmStart,
    pub(crate) certificate: AnchoredContinuationCertificate,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum AnchoredContinuationRefusal {
    EndpointDimensionMismatch {
        anchor_dim: usize,
        target_dim: usize,
    },
    WaypointEvaluationFailed {
        steps: usize,
        waypoint_index: usize,
        reason: String,
    },
    WaypointNotCertified {
        steps: usize,
        waypoint_index: usize,
        inner_converged: bool,
        objective: f64,
    },
    EmptySweep {
        steps: usize,
    },
    EndpointBlockCountMismatch {
        steps: usize,
        coarser_blocks: usize,
        finer_blocks: usize,
    },
    EndpointBlockWidthMismatch {
        steps: usize,
        block_index: usize,
        coarser_width: usize,
        finer_width: usize,
    },
    EndpointCoordinateNotFinite {
        steps: usize,
        block_index: usize,
        coordinate_index: usize,
        coarser: f64,
        finer: f64,
    },
    InvalidInnerTolerance {
        tolerance: f64,
    },
    InvalidEndpointDiscrepancy {
        steps: usize,
        role: &'static str,
        discrepancy: f64,
    },
    /// The ladder spent its whole refinement budget without the endpoint
    /// settling on one mode. Carries the full trail, because the two shapes a
    /// non-settling ladder can have need different repairs and are opposites:
    /// a trail that alternates between `O(1)` and the corrector's floor is a
    /// path oscillating between branches, while one that sits just above the
    /// resolution throughout is a path whose criterion never resolves.
    RefinementBudgetExhausted {
        steps: usize,
        refinements: usize,
        max_refinements: usize,
        criterion_resolution: f64,
        /// `(steps, endpoint discrepancy, criterion agreement)` per refinement.
        trail: Vec<(usize, f64, f64)>,
    },
    StepCountOverflow {
        steps: usize,
    },
    PathResolutionExhausted {
        steps: usize,
        refined_steps: usize,
    },
    HomotopyMemberConstructionFailed {
        steps: usize,
        waypoint_index: usize,
        progress: f64,
        reason: String,
    },
    HomotopyMemberUnavailable {
        steps: usize,
        waypoint_index: usize,
        progress: f64,
    },
}

impl std::fmt::Display for AnchoredContinuationRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EndpointDimensionMismatch {
                anchor_dim,
                target_dim,
            } => write!(
                f,
                "anchor dimension {anchor_dim} differs from target dimension {target_dim}"
            ),
            Self::WaypointEvaluationFailed {
                steps,
                waypoint_index,
                reason,
            } => write!(
                f,
                "{steps}-step sweep failed at waypoint {waypoint_index}: {reason}"
            ),
            Self::WaypointNotCertified {
                steps,
                waypoint_index,
                inner_converged,
                objective,
            } => write!(
                f,
                "{steps}-step sweep waypoint {waypoint_index} was not a certified finite mode \
                 (inner_converged={inner_converged}, objective={objective})"
            ),
            Self::EmptySweep { steps } => {
                write!(f, "{steps}-step sweep produced no endpoint")
            }
            Self::EndpointBlockCountMismatch {
                steps,
                coarser_blocks,
                finer_blocks,
            } => write!(
                f,
                "{steps}-step endpoint has {finer_blocks} blocks but its coarser endpoint has \
                 {coarser_blocks}"
            ),
            Self::EndpointBlockWidthMismatch {
                steps,
                block_index,
                coarser_width,
                finer_width,
            } => write!(
                f,
                "{steps}-step endpoint block {block_index} has width {finer_width} but its \
                 coarser endpoint has width {coarser_width}"
            ),
            Self::EndpointCoordinateNotFinite {
                steps,
                block_index,
                coordinate_index,
                coarser,
                finer,
            } => write!(
                f,
                "{steps}-step endpoint comparison is non-finite at block {block_index}, \
                 coordinate {coordinate_index} (coarser={coarser}, finer={finer})"
            ),
            Self::InvalidInnerTolerance { tolerance } => {
                write!(f, "inner continuation tolerance is invalid ({tolerance})")
            }
            Self::InvalidEndpointDiscrepancy {
                steps,
                role,
                discrepancy,
            } => write!(
                f,
                "{steps}-step {role} endpoint discrepancy is invalid ({discrepancy})"
            ),
            Self::RefinementBudgetExhausted {
                steps,
                refinements,
                max_refinements,
                criterion_resolution,
                trail,
            } => write!(
                f,
                "the continuation endpoint had not settled on one mode after {refinements} of \
                 {max_refinements} refinement(s), the last at {steps} steps: no \
                 {REQUIRED_CONSECUTIVE_AGREEMENTS} consecutive refinements agreed on the \
                 criterion to within its resolution {criterion_resolution:.6e}. Trail \
                 (steps, endpoint discrepancy, criterion agreement): {}",
                trail
                    .iter()
                    .map(|(steps, discrepancy, agreement)| format!(
                        "({steps}, {discrepancy:.6e}, {agreement:.6e})"
                    ))
                    .collect::<Vec<_>>()
                    .join(" "),
            ),
            Self::StepCountOverflow { steps } => {
                write!(
                    f,
                    "doubling the {steps}-step refinement would overflow usize"
                )
            }
            Self::PathResolutionExhausted {
                steps,
                refined_steps,
            } => write!(
                f,
                "endpoint still moves at {steps} steps, but {refined_steps} steps are below the \
                 continuation path's floating-point resolution"
            ),
            Self::HomotopyMemberConstructionFailed {
                steps,
                waypoint_index,
                progress,
                reason,
            } => write!(
                f,
                "{steps}-step coefficient-objective homotopy failed to construct waypoint \
                 {waypoint_index} at progress {progress:.17}: {reason}"
            ),
            Self::HomotopyMemberUnavailable {
                steps,
                waypoint_index,
                progress,
            } => write!(
                f,
                "{steps}-step coefficient-objective homotopy omitted waypoint {waypoint_index} \
                 at progress {progress:.17} after declaring an anchor"
            ),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum ContinuationRefinement {
    Certified(AnchoredContinuationCertificate),
    Refine,
}

/// Everything one refinement of the ladder knows about itself.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ContinuationRefinementReading {
    pub(crate) steps: usize,
    /// State-space distance between this endpoint and the coarser one.
    pub(crate) discrepancy: f64,
    /// The same for the previous refinement, if there was one.
    pub(crate) previous_discrepancy: Option<f64>,
    /// Relative distance between the two endpoints' criterion values.
    pub(crate) criterion_agreement: f64,
    /// How many consecutive refinements — INCLUDING this one — agreed.
    pub(crate) consecutive_agreements: usize,
}

/// The verdict on one refinement, given the resolutions it is judged against.
///
/// The certifying quantity is `criterion_agreement`, not `discrepancy`. Both are
/// carried, because the certificate reports both.
pub(crate) fn continuation_refinement_decision(
    reading: ContinuationRefinementReading,
    inner_tolerance: f64,
    criterion_resolution: f64,
) -> Result<ContinuationRefinement, AnchoredContinuationRefusal> {
    let ContinuationRefinementReading {
        steps,
        discrepancy,
        previous_discrepancy,
        criterion_agreement,
        consecutive_agreements,
    } = reading;
    if !inner_tolerance.is_finite() || inner_tolerance < 0.0 {
        return Err(AnchoredContinuationRefusal::InvalidInnerTolerance {
            tolerance: inner_tolerance,
        });
    }
    if !criterion_resolution.is_finite() || criterion_resolution < 0.0 {
        return Err(AnchoredContinuationRefusal::InvalidInnerTolerance {
            tolerance: criterion_resolution,
        });
    }
    if !discrepancy.is_finite() || discrepancy < 0.0 {
        return Err(AnchoredContinuationRefusal::InvalidEndpointDiscrepancy {
            steps,
            role: "current",
            discrepancy,
        });
    }
    if !criterion_agreement.is_finite() || criterion_agreement < 0.0 {
        return Err(AnchoredContinuationRefusal::InvalidEndpointDiscrepancy {
            steps,
            role: "criterion",
            discrepancy: criterion_agreement,
        });
    }
    let observed_contraction_factor = match previous_discrepancy {
        Some(previous) => {
            if !previous.is_finite() || previous < 0.0 {
                return Err(AnchoredContinuationRefusal::InvalidEndpointDiscrepancy {
                    steps,
                    role: "previous",
                    discrepancy: previous,
                });
            }
            Some(if previous == 0.0 {
                if discrepancy == 0.0 {
                    0.0
                } else {
                    f64::INFINITY
                }
            } else {
                discrepancy / previous
            })
        }
        None => None,
    };
    if consecutive_agreements >= REQUIRED_CONSECUTIVE_AGREEMENTS {
        return Ok(ContinuationRefinement::Certified(
            AnchoredContinuationCertificate {
                steps,
                endpoint_discrepancy: discrepancy,
                inner_tolerance,
                observed_contraction_factor,
                criterion_agreement,
                criterion_resolution,
                consecutive_agreements,
            },
        ));
    }
    Ok(ContinuationRefinement::Refine)
}

/// The record a declined #2661 continuation leaves on the fit: the mode is the
/// one the caller's seed reached, for the reason the continuation refused.
pub(crate) fn declined_continuation_selection(
    refusal: &AnchoredContinuationRefusal,
) -> gam_solve::model_types::CoefficientModeSelection {
    gam_solve::model_types::CoefficientModeSelection::SeedSelected {
        reason: refusal.to_string(),
    }
}

pub(crate) fn anchored_continuation_seed<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
    layout: &PenaltyLabelLayout,
    rho_prior: &gam_problem::RhoPrior,
    rho_anchor: &Array1<f64>,
    rho_target: &Array1<f64>,
) -> Result<CertifiedAnchoredContinuationSeed, AnchoredContinuationRefusal> {
    if rho_anchor.len() != rho_target.len() {
        return Err(AnchoredContinuationRefusal::EndpointDimensionMismatch {
            anchor_dim: rho_anchor.len(),
            target_dim: rho_target.len(),
        });
    }
    let path = ContinuationPath {
        family,
        specs,
        options,
        layout,
        rho_prior,
        rho_anchor,
        rho_target,
        anchor_mode: std::cell::OnceCell::new(),
    };
    certify_refined_continuation(&path, options, false)
}

/// One sweep's endpoint, together with the value of the criterion the seed
/// exists to make well defined, evaluated at it.
///
/// The pair travels together because the ladder judges endpoints by the
/// criterion and reports them by their state; separating them is how one of them
/// gets compared at the wrong ρ.
pub(crate) struct SweptEndpoint {
    pub(crate) warm_start: ConstrainedWarmStart,
    pub(crate) criterion_value: f64,
}

pub(crate) trait RefinedContinuationPath {
    fn sweep(&self, steps: usize) -> Result<SweptEndpoint, AnchoredContinuationRefusal>;
    fn resolves_steps(&self, refined_steps: usize) -> bool;
    fn endpoint_discrepancy(
        &self,
        steps: usize,
        coarser: &ConstrainedWarmStart,
        finer: &ConstrainedWarmStart,
    ) -> Result<f64, AnchoredContinuationRefusal>;
    /// What this path is a continuation IN, for the refinement trail below.
    fn label(&self) -> &'static str;
}

/// Relative distance between two criterion values, on the same scale-free form
/// the state discrepancy uses so the two are read in the same units.
fn criterion_agreement(coarser: f64, finer: f64) -> f64 {
    if !coarser.is_finite() || !finer.is_finite() {
        return f64::INFINITY;
    }
    (coarser - finer).abs() / (1.0 + coarser.abs().max(finer.abs()))
}

/// How many dyadic refinements the ladder may spend.
///
/// #2661 established the requirement this answers: accepting arbitrarily slow
/// progress makes the loop operationally unbounded, and each refinement DOUBLES
/// the number of full corrector solves. That is a statement about a resource, so
/// it is bounded as one rather than smuggled into a convergence ratio (which is
/// what it was, and which #2612 measured cannot read this ladder — see
/// [`certify_refined_continuation`]).
///
/// The bound is derived, not chosen: **the seed may not cost more correctors
/// than the outer search it seeds is budgeted for.** A ladder through `D`
/// refinements runs `1 + 2 + 4 + … + 2^D = 2^{D+1} − 1` correctors, and the outer
/// search is allowed `outer_max_iter` iterations each of which pays at least one,
/// so `2^{D+1} ≤ outer_max_iter`, i.e. `D = ⌊log₂(outer_max_iter)⌋ − 1`.
///
/// Floored at `REQUIRED_CONSECUTIVE_AGREEMENTS + 1`, which is the fewest
/// refinements that can produce a verdict at all: a budget below it refuses
/// every path regardless of what the path does, which is a disablement rather
/// than a budget.
pub(crate) fn continuation_refinement_budget(outer_max_iter: usize) -> usize {
    let from_outer_budget = usize::BITS
        .saturating_sub(outer_max_iter.leading_zeros())
        .saturating_sub(2) as usize;
    from_outer_budget.max(REQUIRED_CONSECUTIVE_AGREEMENTS + 1)
}

/// Refine the continuation's discretization until its endpoint settles on one
/// mode, and certify that it did.
///
/// # The endpoint sequence is MODE-VALUED, and everything here follows from it
///
/// Every sweep's LAST waypoint corrects at the target family and the target ρ,
/// to the inner solver's own KKT tolerance. So each endpoint is an *exact* mode
/// of the *same* function — not a discretization-perturbed approximation of one.
/// Refining the path does not shrink an error; it changes WHICH mode the path
/// arrives at. The measured trail on the #2612 penguins stride-4 armed refit
/// says exactly that:
///
/// ```text
///   steps  1 -> 2    endpoint discrepancy 1.521120e0
///   steps  2 -> 4                         3.328619e-5
///   steps  4 -> 8                         1.695145e0
///   steps  8 -> 16                        5.413481e-5
/// ```
///
/// Two values, four orders apart, alternating: `O(1)` when the two sweeps land
/// on different modes, `O(5e-5)` — the corrector's own reproducibility — when
/// they land on the same one. There is no `O(hᑫ)` decay to observe.
///
/// ## Consequence 1: the contraction premise cannot read this ladder
///
/// `d_k ≤ ½ d_{k−1}` is the signature of a smoothly convergent discretization.
/// On the trail above it fires at `steps = 8` — the FIRST refinement that
/// actually tracks the branch — using as its baseline the `3.328619e-5`
/// agreement of two coarse sweeps that had both jumped to the same wrong mode.
/// The premise is retained only as reported evidence
/// ([`AnchoredContinuationCertificate::observed_contraction_factor`]).
///
/// ## Consequence 2: one agreement is not evidence of a limit
///
/// The `2 → 4` agreement above is a full agreement on a mode the next refinement
/// leaves. [`REQUIRED_CONSECUTIVE_AGREEMENTS`] carries that reasoning.
///
/// ## Consequence 3: the yardstick has to be the criterion, not the coefficients
///
/// The old bar compared a relative sup-norm over linear predictors against
/// `options.inner_tol` — a KKT-RESIDUAL tolerance. Those are different
/// quantities, and on this fixture the mismatch is fatal rather than cosmetic:
/// two sweeps reaching the SAME mode differ by `3.3e-5` and `5.4e-5` in that
/// norm against a `1e-5` bar, so "the same mode, reached twice" could not be
/// certified at any refinement depth. The reason is physical: the armed mode is
/// nearly flat in one direction (`λ_min ≈ 4.7e-7` on this fixture), so `β̂` is
/// poorly determined by a KKT residual while the criterion built from it is
/// well determined — the same two endpoints agree to `5.0e-7` in the criterion.
///
/// And the criterion is what the seed exists for. `V(ρ) = ℓ_p(θ̂(ρ), ρ)` is a
/// function of ρ only once a selection rule fixes `θ̂`; the rule is this
/// continuation; so two endpoints that the criterion cannot tell apart are the
/// same seed for every purpose the caller has. The bar is therefore the outer
/// solver's own relative-cost resolution, in the criterion's own units.
///
/// The state discrepancy is still computed and still reported — it is what makes
/// a branch change visible in the trail — it simply is not the verdict.
pub(crate) fn certify_refined_continuation<P: RefinedContinuationPath>(
    path: &P,
    options: &BlockwiseFitOptions,
    refine_uncertified_waypoints: bool,
) -> Result<CertifiedAnchoredContinuationSeed, AnchoredContinuationRefusal> {
    let inner_tolerance = options.inner_tol;
    // The outer solver's own relative-cost resolution: two criterion values
    // closer than this are values the search that consumes them cannot separate.
    // `outer_rel_cost_tol` is the relative one where a family sets it;
    // `outer_tol` is the fallback, and it is the same quantity the outer
    // convergence test is denominated in.
    let criterion_resolution = options.outer_rel_cost_tol.unwrap_or(options.outer_tol);
    let max_refinements = continuation_refinement_budget(options.outer_max_iter);
    let mut coarser: Option<SweptEndpoint> = None;
    let mut previous_discrepancy: Option<f64> = None;
    let mut consecutive_agreements = 0usize;
    let mut refinements = 0usize;
    let mut trail: Vec<(usize, f64, f64)> = Vec::new();
    let mut steps = 1usize;
    loop {
        let endpoint = match path.sweep(steps) {
            Ok(endpoint) => endpoint,
            Err(
                refusal @ AnchoredContinuationRefusal::WaypointNotCertified {
                    waypoint_index, ..
                },
            ) if refine_uncertified_waypoints && waypoint_index > 0 => {
                let refined = steps
                    .checked_mul(2)
                    .ok_or(AnchoredContinuationRefusal::StepCountOverflow { steps })?;
                if !path.resolves_steps(refined) {
                    return Err(refusal);
                }
                // Charged against the same budget as a comparison refinement:
                // it doubles the same corrector count, so exempting it would
                // reopen the unbounded loop by another door.
                if refinements >= max_refinements {
                    return Err(AnchoredContinuationRefusal::RefinementBudgetExhausted {
                        steps,
                        refinements,
                        max_refinements,
                        criterion_resolution,
                        trail,
                    });
                }
                refinements += 1;
                log::debug!(
                    "[OUTER] {} continuation refining {steps}→{refined} steps after waypoint \
                     {waypoint_index} did not certify",
                    path.label(),
                );
                steps = refined;
                continue;
            }
            Err(refusal) => return Err(refusal),
        };
        if let Some(previous) = coarser.as_ref() {
            let discrepancy =
                path.endpoint_discrepancy(steps, &previous.warm_start, &endpoint.warm_start)?;
            let agreement =
                criterion_agreement(previous.criterion_value, endpoint.criterion_value);
            if agreement <= criterion_resolution {
                consecutive_agreements += 1;
            } else {
                consecutive_agreements = 0;
            }
            trail.push((steps, discrepancy, agreement));
            // The refinement TRAIL, not just the pair the verdict is taken on.
            // A mode-valued ladder's discrepancy sequence alternates between two
            // scales; printing one ratio out of it reads as a convergence rate
            // and is not one.
            log::debug!(
                "[OUTER] {} continuation refinement: steps={steps} discrepancy={discrepancy:.6e} \
                 previous={} criterion={:.9e} -> {:.9e} (agreement={agreement:.6e} vs resolution \
                 {criterion_resolution:.6e}, consecutive={consecutive_agreements}/{}) \
                 inner_tolerance={inner_tolerance:.6e}",
                path.label(),
                previous_discrepancy
                    .map(|value| format!("{value:.6e}"))
                    .unwrap_or_else(|| "none".to_string()),
                previous.criterion_value,
                endpoint.criterion_value,
                REQUIRED_CONSECUTIVE_AGREEMENTS,
            );
            match continuation_refinement_decision(
                ContinuationRefinementReading {
                    steps,
                    discrepancy,
                    previous_discrepancy,
                    criterion_agreement: agreement,
                    consecutive_agreements,
                },
                inner_tolerance,
                criterion_resolution,
            )? {
                ContinuationRefinement::Certified(certificate) => {
                    return Ok(CertifiedAnchoredContinuationSeed {
                        warm_start: endpoint.warm_start,
                        certificate,
                    });
                }
                ContinuationRefinement::Refine => {}
            }
            previous_discrepancy = Some(discrepancy);
        }
        let refined = steps
            .checked_mul(2)
            .ok_or(AnchoredContinuationRefusal::StepCountOverflow { steps })?;
        if !path.resolves_steps(refined) {
            return Err(AnchoredContinuationRefusal::PathResolutionExhausted {
                steps,
                refined_steps: refined,
            });
        }
        // `refinements` counts refinements PERFORMED, so the check comes before
        // the increment and the refusal reports what was actually spent.
        if refinements >= max_refinements {
            return Err(AnchoredContinuationRefusal::RefinementBudgetExhausted {
                steps,
                refinements,
                max_refinements,
                criterion_resolution,
                trail,
            });
        }
        refinements += 1;
        coarser = Some(endpoint);
        steps = refined;
    }
}

/// The segment in ρ that the continuation follows, together with everything
/// needed to solve at a point on it.
///
/// This exists so a sweep is `path.sweep(steps)` rather than an eight-argument
/// call: the seven parts are one object — a path — not seven independent knobs,
/// and naming it that way keeps the endpoints from being transposed.
struct ContinuationPath<'a, F> {
    family: &'a F,
    specs: &'a [ParameterBlockSpec],
    options: &'a BlockwiseFitOptions,
    layout: &'a PenaltyLabelLayout,
    rho_prior: &'a gam_problem::RhoPrior,
    rho_anchor: &'a Array1<f64>,
    rho_target: &'a Array1<f64>,
    /// The anchor waypoint's corrected mode, once a sweep has solved it.
    anchor_mode: std::cell::OnceCell<AnchorWaypointMode>,
}

/// The corrected mode at the continuation's anchor, kept for the ladder's
/// later sweeps (gam#2928).
///
/// Every sweep's first corrector solves the inner problem at `ρ_A` from the
/// caller's coefficients, and the mode there is unique by construction (see
/// [`ContinuationPath::sweep`]). Each refinement therefore repeated one
/// deterministic computation on the same inputs: 3 of a 4-step ladder's 13
/// inner solves, 21 s of a 125 s fit at n = 300,000. The ladder now solves it
/// once. Every later waypoint is still corrected afresh in each refinement, so
/// the refinements stay independent wherever a branch can be chosen.
///
/// The solve reads the family, the options, the layout, `ρ_A` and the specs,
/// whose `initial_beta` are the caller's coefficients it starts from (a cold
/// start, `buildblock_states`). Every one of them is a shared borrow of the
/// path and cannot change under it. The mode is still keyed by `ρ_A`'s bits,
/// so an anchor that does not match them is solved, never read.
pub(crate) struct AnchorWaypointMode {
    rho_bits: Vec<u64>,
    mode: ConstrainedWarmStart,
}

impl AnchorWaypointMode {
    pub(crate) fn new(rho: &Array1<f64>, mode: ConstrainedWarmStart) -> Self {
        Self {
            rho_bits: rho.iter().map(|value| value.to_bits()).collect(),
            mode,
        }
    }

    /// The kept mode, only for the anchor it was solved at.
    pub(crate) fn at(&self, rho: &Array1<f64>) -> Option<&ConstrainedWarmStart> {
        (self.rho_bits.len() == rho.len()
            && self
                .rho_bits
                .iter()
                .zip(rho.iter())
                .all(|(&bits, value)| bits == value.to_bits()))
        .then_some(&self.mode)
    }
}

impl<F: CustomFamily + Clone + Send + Sync + 'static> ContinuationPath<'_, F> {
    /// The corrected mode at `ρ_A` from the caller's coefficients: solved by
    /// the first sweep, read by the rest.
    fn anchor_waypoint_mode(
        &self,
        steps: usize,
    ) -> Result<ConstrainedWarmStart, AnchoredContinuationRefusal> {
        if let Some(mode) = self
            .anchor_mode
            .get()
            .and_then(|kept| kept.at(self.rho_anchor))
        {
            return Ok(mode.clone());
        }
        let (_, warm_start) = correct_labeled_coefficient_mode(
            self.family,
            self.specs,
            self.options,
            self.layout,
            self.rho_anchor,
            None,
        )
        .map_err(
            |error| AnchoredContinuationRefusal::WaypointEvaluationFailed {
                steps,
                waypoint_index: 0,
                reason: error.to_string(),
            },
        )?;
        self.anchor_mode
            .get_or_init(|| AnchorWaypointMode::new(self.rho_anchor, warm_start.clone()));
        Ok(warm_start)
    }
}

impl<F: CustomFamily + Clone + Send + Sync + 'static> RefinedContinuationPath
    for ContinuationPath<'_, F>
{
    /// One continuation sweep at a fixed discretization: solve the inner problem
    /// at each waypoint, carrying the previous mode forward as the predictor.
    ///
    /// The sweep starts AT the anchor (`step == 0`). That first solve is the one
    /// that makes the whole construction well defined: it is the only corrector
    /// that runs from the caller's coefficients, and at the anchor every
    /// penalized term sits on its penalty nullspace, so the surviving problem
    /// has a single mode and the caller's seed cannot select anything. Every
    /// later waypoint is warm-started from the previous one, so the branch is
    /// carried forward rather than rediscovered. Because that first solve is the
    /// same computation in every sweep, it runs once per ladder
    /// ([`AnchorWaypointMode`]).
    ///
    /// The final waypoint uses `rho_target` verbatim rather than the
    /// reconstructed `ρ_A + 1·(ρ − ρ_A)`: the endpoint must be the mode *at the
    /// requested ρ* bitwise, because everything downstream binds the mode and
    /// its ρ as one identity.
    fn sweep(&self, steps: usize) -> Result<SweptEndpoint, AnchoredContinuationRefusal> {
        let mut carried: Option<ConstrainedWarmStart> = None;
        for step in 0..=steps {
            if step == 0 && step < steps {
                carried = Some(self.anchor_waypoint_mode(steps)?);
                continue;
            }
            let waypoint = if step == steps {
                self.rho_target.clone()
            } else {
                let t = step as f64 / steps as f64;
                Array1::from_shape_fn(self.rho_target.len(), |j| {
                    self.rho_anchor[j] + t * (self.rho_target[j] - self.rho_anchor[j])
                })
            };
            let (inner, warm_start) = correct_labeled_coefficient_mode(
                self.family,
                self.specs,
                self.options,
                self.layout,
                &waypoint,
                carried.as_ref(),
            )
            .map_err(|error| {
                AnchoredContinuationRefusal::WaypointEvaluationFailed {
                    steps,
                    waypoint_index: step,
                    reason: error.to_string(),
                }
            })?;
            if step < steps {
                carried = Some(warm_start);
                continue;
            }
            let eval = outerobjective_from_coefficient_mode_labeled(
                self.family,
                self.specs,
                self.options,
                self.layout,
                &waypoint,
                self.rho_prior,
                inner,
                EvalMode::ValueOnly,
            )
            .map_err(|error| {
                AnchoredContinuationRefusal::WaypointEvaluationFailed {
                    steps,
                    waypoint_index: step,
                    reason: error.to_string(),
                }
            })?;
            if !eval.inner_converged || !eval.objective.is_finite() {
                return Err(AnchoredContinuationRefusal::WaypointNotCertified {
                    steps,
                    waypoint_index: step,
                    inner_converged: eval.inner_converged,
                    objective: eval.objective,
                });
            }
            return Ok(SweptEndpoint {
                warm_start: eval.warm_start,
                criterion_value: eval.objective,
            });
        }
        Err(AnchoredContinuationRefusal::EmptySweep { steps })
    }

    fn resolves_steps(&self, refined_steps: usize) -> bool {
        continuation_path_resolves_steps(self.rho_anchor, self.rho_target, refined_steps)
    }

    fn endpoint_discrepancy(
        &self,
        steps: usize,
        coarser: &ConstrainedWarmStart,
        finer: &ConstrainedWarmStart,
    ) -> Result<f64, AnchoredContinuationRefusal> {
        continuation_endpoint_discrepancy(steps, coarser, finer)
    }

    fn label(&self) -> &'static str {
        "#2661 anchored"
    }
}

/// Follow a family-declared coefficient-objective homotopy at one fixed `ρ`.
///
/// The zero member must have a uniquely selected coefficient mode; each
/// waypoint is then corrected exactly and carried into the next member. The
/// target waypoint always uses `family` itself, so the seed is bitwise attached
/// to the production objective rather than to a reconstructed approximation at
/// progress one.
pub(crate) fn coefficient_objective_homotopy_seed<
    F: CustomFamily + Clone + Send + Sync + 'static,
>(
    family: &F,
    specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
    layout: &PenaltyLabelLayout,
    rho_prior: &gam_problem::RhoPrior,
    rho: &Array1<f64>,
) -> Result<Option<CertifiedAnchoredContinuationSeed>, AnchoredContinuationRefusal> {
    match family.coefficient_mode_homotopy_member(0.0) {
        Ok(Some(_)) => {}
        Ok(None) => return Ok(None),
        Err(reason) => {
            return Err(
                AnchoredContinuationRefusal::HomotopyMemberConstructionFailed {
                    steps: 0,
                    waypoint_index: 0,
                    progress: 0.0,
                    reason,
                },
            );
        }
    }
    let path = CoefficientObjectiveHomotopyPath {
        family,
        specs,
        options,
        layout,
        rho_prior,
        rho,
    };
    certify_refined_continuation(&path, options, true).map(Some)
}

struct CoefficientObjectiveHomotopyPath<'a, F> {
    family: &'a F,
    specs: &'a [ParameterBlockSpec],
    options: &'a BlockwiseFitOptions,
    layout: &'a PenaltyLabelLayout,
    rho_prior: &'a gam_problem::RhoPrior,
    rho: &'a Array1<f64>,
}

impl<F: CustomFamily + Clone + Send + Sync + 'static> RefinedContinuationPath
    for CoefficientObjectiveHomotopyPath<'_, F>
{
    fn sweep(&self, steps: usize) -> Result<SweptEndpoint, AnchoredContinuationRefusal> {
        let mut carried: Option<ConstrainedWarmStart> = None;
        for step in 0..=steps {
            let progress = step as f64 / steps as f64;
            let member = if step == steps {
                self.family.clone()
            } else {
                self.family
                    .coefficient_mode_homotopy_member(progress)
                    .map_err(|reason| {
                        AnchoredContinuationRefusal::HomotopyMemberConstructionFailed {
                            steps,
                            waypoint_index: step,
                            progress,
                            reason,
                        }
                    })?
                    .ok_or(AnchoredContinuationRefusal::HomotopyMemberUnavailable {
                        steps,
                        waypoint_index: step,
                        progress,
                    })?
            };
            let (inner, warm_start) = correct_labeled_coefficient_mode(
                &member,
                self.specs,
                self.options,
                self.layout,
                self.rho,
                carried.as_ref(),
            )
            .map_err(|error| {
                AnchoredContinuationRefusal::WaypointEvaluationFailed {
                    steps,
                    waypoint_index: step,
                    reason: error.to_string(),
                }
            })?;
            // The per-waypoint trail. The endpoint discrepancy compares two
            // sweeps at their LAST waypoint only, so a path that changed branch
            // partway is indistinguishable from one that landed differently by
            // accumulation. `|eta|inf` is the coordinate the discrepancy is
            // taken in, so the two readings are the same instrument.
            log::debug!(
                "[OUTER] coefficient-objective homotopy: steps={steps} waypoint={step} \
                 progress={progress:.6} inner_merit={:.9e} |eta|inf={:.6e} \
                 inner(loglik={} penalty={} cycles={} converged={})",
                -inner.log_likelihood + inner.penalty_value,
                waypoint_eta_sup_norm(self.specs, &warm_start),
                inner.log_likelihood,
                inner.penalty_value,
                inner.cycles,
                inner.converged,
            );
            if step < steps {
                carried = Some(warm_start);
                continue;
            }
            let eval = outerobjective_from_coefficient_mode_labeled(
                &member,
                self.specs,
                self.options,
                self.layout,
                self.rho,
                self.rho_prior,
                inner,
                EvalMode::ValueOnly,
            )
            .map_err(|error| {
                AnchoredContinuationRefusal::WaypointEvaluationFailed {
                    steps,
                    waypoint_index: step,
                    reason: error.to_string(),
                }
            })?;
            if !eval.inner_converged || !eval.objective.is_finite() {
                return Err(AnchoredContinuationRefusal::WaypointNotCertified {
                    steps,
                    waypoint_index: step,
                    inner_converged: eval.inner_converged,
                    objective: eval.objective,
                });
            }
            return Ok(SweptEndpoint {
                warm_start: eval.warm_start,
                criterion_value: eval.objective,
            });
        }
        Err(AnchoredContinuationRefusal::EmptySweep { steps })
    }

    fn resolves_steps(&self, refined_steps: usize) -> bool {
        1.0 / refined_steps as f64 > f64::EPSILON
    }

    fn endpoint_discrepancy(
        &self,
        steps: usize,
        coarser: &ConstrainedWarmStart,
        finer: &ConstrainedWarmStart,
    ) -> Result<f64, AnchoredContinuationRefusal> {
        coefficient_objective_endpoint_discrepancy(steps, self.specs, coarser, finer)
    }

    fn label(&self) -> &'static str {
        "coefficient-objective"
    }
}

/// Compare two coefficient modes in the coordinates the model objective can
/// observe: fitted linear predictors plus the cached likelihood and penalty
/// values at the common target family/rho.
///
/// Raw coefficient distance is not invariant to basis scaling and treats drift
/// along a design-null or nearly-null direction as a different mode even when
/// the family and penalty see the same fitted function. That is exactly the
/// quasi-separated multinomial geometry for which the Jeffreys homotopy is
/// needed. The solver design (including stacked multi-channel designs) is the
/// authoritative map from coefficients to family state, while the cached
/// likelihood/penalty pair catches any objective-relevant penalty motion not
/// visible in eta.
/// `‖η‖∞` over every block of a waypoint, in the same linear-predictor
/// coordinates [`coefficient_objective_endpoint_discrepancy`] compares. Purely
/// diagnostic; a block whose design cannot be applied contributes nothing rather
/// than turning a log line into a failure.
fn waypoint_eta_sup_norm(specs: &[ParameterBlockSpec], warm: &ConstrainedWarmStart) -> f64 {
    let mut worst = 0.0_f64;
    for (beta, spec) in warm.block_beta.iter().zip(specs.iter()) {
        if beta.len() != spec.solver_design().ncols() {
            continue;
        }
        let eta = spec.solver_design().matrixvectormultiply(beta) + spec.solver_offset();
        for value in eta.iter() {
            worst = worst.max(value.abs());
        }
    }
    worst
}

fn coefficient_objective_endpoint_discrepancy(
    steps: usize,
    specs: &[ParameterBlockSpec],
    coarser: &ConstrainedWarmStart,
    finer: &ConstrainedWarmStart,
) -> Result<f64, AnchoredContinuationRefusal> {
    if coarser.block_beta.len() != finer.block_beta.len() {
        return Err(AnchoredContinuationRefusal::EndpointBlockCountMismatch {
            steps,
            coarser_blocks: coarser.block_beta.len(),
            finer_blocks: finer.block_beta.len(),
        });
    }
    if coarser.block_beta.len() != specs.len() {
        return Err(AnchoredContinuationRefusal::EndpointBlockCountMismatch {
            steps,
            coarser_blocks: coarser.block_beta.len(),
            finer_blocks: specs.len(),
        });
    }
    let mut worst = 0.0_f64;
    for (block_index, ((a, b), spec)) in coarser
        .block_beta
        .iter()
        .zip(finer.block_beta.iter())
        .zip(specs.iter())
        .enumerate()
    {
        if a.len() != b.len() {
            return Err(AnchoredContinuationRefusal::EndpointBlockWidthMismatch {
                steps,
                block_index,
                coarser_width: a.len(),
                finer_width: b.len(),
            });
        }
        let eta_a = spec.solver_design().matrixvectormultiply(a) + spec.solver_offset();
        let eta_b = spec.solver_design().matrixvectormultiply(b) + spec.solver_offset();
        for (coordinate_index, (x, y)) in eta_a.iter().zip(eta_b.iter()).enumerate() {
            if !x.is_finite() || !y.is_finite() {
                return Err(AnchoredContinuationRefusal::EndpointCoordinateNotFinite {
                    steps,
                    block_index,
                    coordinate_index,
                    coarser: *x,
                    finer: *y,
                });
            }
            worst = worst.max((x - y).abs() / (1.0 + x.abs().max(y.abs())));
        }
    }
    if let (Some(a), Some(b)) = (&coarser.cached_inner, &finer.cached_inner) {
        for (x, y) in [
            (a.log_likelihood, b.log_likelihood),
            (a.penalty_value, b.penalty_value),
        ] {
            if !x.is_finite() || !y.is_finite() {
                return Err(AnchoredContinuationRefusal::InvalidEndpointDiscrepancy {
                    steps,
                    role: "objective-state",
                    discrepancy: f64::NAN,
                });
            }
            worst = worst.max((x - y).abs() / (1.0 + x.abs().max(y.abs())));
        }
    }
    Ok(worst)
}

/// How far apart two continuation endpoints are, relative to each coefficient's
/// own magnitude.
///
/// The measure is scale-free so it is invariant to the units of a block, and it
/// is compared against the inner solve's own convergence tolerance rather than
/// against a separate constant: two endpoints closer than what the corrector
/// itself resolves are not evidence that the discretization still matters.
/// Endpoints with different structures or non-finite coordinates produce a
/// typed refusal, because they cannot furnish a meaningful discrepancy.
fn continuation_endpoint_discrepancy(
    steps: usize,
    coarser: &ConstrainedWarmStart,
    finer: &ConstrainedWarmStart,
) -> Result<f64, AnchoredContinuationRefusal> {
    if coarser.block_beta.len() != finer.block_beta.len() {
        return Err(AnchoredContinuationRefusal::EndpointBlockCountMismatch {
            steps,
            coarser_blocks: coarser.block_beta.len(),
            finer_blocks: finer.block_beta.len(),
        });
    }
    let mut worst = 0.0_f64;
    for (block_index, (a, b)) in coarser
        .block_beta
        .iter()
        .zip(finer.block_beta.iter())
        .enumerate()
    {
        if a.len() != b.len() {
            return Err(AnchoredContinuationRefusal::EndpointBlockWidthMismatch {
                steps,
                block_index,
                coarser_width: a.len(),
                finer_width: b.len(),
            });
        }
        for (coordinate_index, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            if !x.is_finite() || !y.is_finite() {
                return Err(AnchoredContinuationRefusal::EndpointCoordinateNotFinite {
                    steps,
                    block_index,
                    coordinate_index,
                    coarser: *x,
                    finer: *y,
                });
            }
            worst = worst.max((x - y).abs() / (1.0 + x.abs().max(y.abs())));
        }
    }
    Ok(worst)
}

/// Whether a `steps`-way uniform split of the continuation path is still
/// resolvable in floating point.
///
/// Once a step is smaller than the spacing of the ρ values it separates, the
/// waypoints collapse onto each other and refining further cannot change the
/// endpoint — the loop must stop rather than spin.
fn continuation_path_resolves_steps(
    rho_anchor: &Array1<f64>,
    rho_target: &Array1<f64>,
    steps: usize,
) -> bool {
    rho_anchor
        .iter()
        .zip(rho_target.iter())
        .any(|(anchor, target)| {
            let span = (target - anchor).abs();
            let scale = anchor.abs().max(target.abs()).max(1.0);
            span / steps as f64 > f64::EPSILON * scale
        })
}

/// Bind one evaluator-owned coefficient mode to the optimizer-owned terminal
/// certificate and consume the carrier on success.
///
/// Every comparison is bitwise. Numerically close state is not interchangeable
/// provenance for a nonconvex profiled objective, and there is deliberately no
/// warm-start/re-evaluation fallback when any part of the identity differs.
pub(crate) fn bind_certified_custom_family_terminal_mode(
    terminal: CustomFamilyTerminalMode,
    certified_outer: &gam_solve::rho_optimizer::CertifiedOuterResult,
) -> Result<CustomFamilyOwnedMode, CustomFamilyError> {
    let certified_gradient = certified_outer.final_gradient().ok_or_else(|| {
        CustomFamilyError::Optimization {
            context: "fit_custom_family terminal gradient ownership",
            reason: "certified outer result retained no exact analytic terminal gradient; no fit was assembled"
                .to_string(),
        }
    })?;
    if terminal.theta.len() != certified_outer.rho().len()
        || terminal
            .theta
            .iter()
            .zip(certified_outer.rho().iter())
            .any(|(terminal, certified)| terminal.to_bits() != certified.to_bits())
    {
        return Err(CustomFamilyError::InvalidInput {
            context: "fit_custom_family terminal theta identity",
            reason: format!(
                "terminal coefficient mode does not bitwise match the certified outer \
                 hyperparameter vector: terminal.theta={:?} vs certified.rho={:?}",
                terminal.theta.as_slice().unwrap_or(&[]),
                certified_outer.rho().as_slice().unwrap_or(&[]),
            ),
        });
    }
    if terminal.objective.to_bits() != certified_outer.final_value().to_bits() {
        return Err(CustomFamilyError::Optimization {
            context: "fit_custom_family terminal objective identity",
            reason: format!(
                "terminal coefficient-mode objective does not bitwise match the certified outer objective: terminal={:.17e}, certified={:.17e}",
                terminal.objective,
                certified_outer.final_value(),
            ),
        });
    }
    if terminal.gradient.len() != certified_gradient.len()
        || terminal
            .gradient
            .iter()
            .zip(certified_gradient.iter())
            .any(|(terminal, certified)| terminal.to_bits() != certified.to_bits())
    {
        return Err(CustomFamilyError::Optimization {
            context: "fit_custom_family terminal gradient identity",
            reason: "terminal coefficient-mode gradient does not bitwise match the optimizer-owned analytic certificate gradient"
                .to_string(),
        });
    }
    if terminal.mode.objective.to_bits() != terminal.objective.to_bits()
        || terminal.mode.rho.len() != terminal.theta.len()
        || terminal
            .mode
            .rho
            .iter()
            .zip(terminal.theta.iter())
            .any(|(mode, terminal)| mode.to_bits() != terminal.to_bits())
    {
        return Err(CustomFamilyError::Optimization {
            context: "fit_custom_family terminal carrier identity",
            reason: "terminal outer payload and its owned coefficient mode have different objective or hyperparameter bits"
                .to_string(),
        });
    }
    Ok(terminal.mode)
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
    log::debug!(
        "[STAGE] identifiability canonicalise: start blocks={} n={} p_total_raw={}",
        raw_specs.len(),
        canonical_n_rows,
        canonical_n_cols_raw,
    );
    let canonical =
        gam_identifiability::canonical::canonicalize_for_identifiability_with_operating_scalars(
            raw_specs,
            &pre_fit_coefficient_coordinates(family, raw_specs),
            pre_fit_operating_scalars(family, raw_specs)?,
        )?;
    let canonical_n_cols_red: usize = canonical
        .reduced_specs
        .iter()
        .map(|s| s.design.ncols())
        .sum();
    log::debug!(
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
        log::debug!("[identifiability audit] {}", canonical.audit.summary);
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
            log::debug!(
                "[identifiability audit] alias-cluster {a} ~ {b}: {count} direction-pair{plural} \
                 (overlap {min:.4}..{max:.4}; {near_one} ≥0.9999)",
                plural = if count == 1 { "" } else { "s" },
            );
        }
        if log::log_enabled!(log::Level::Trace) {
            for ((a, b), pairs) in &by_pair {
                let mut sorted = pairs.clone();
                sorted.sort_by(|p, q| {
                    q.overlap
                        .partial_cmp(&p.overlap)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                for pair in sorted.iter().take(3) {
                    log::trace!(
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
        log::debug!(
            "[identifiability audit] dropped: block='{}' local_col={} ({})",
            drop.block,
            drop.column,
            drop.reason,
        );
    }
    let specs: &[ParameterBlockSpec] = &canonical.reduced_specs;
    let penalty_counts = validate_blockspecs(specs)?;
    crate::inner_blockwise_fit::refuse_non_finite_declared_joint_curvature(family, specs)?;

    // gam#1587: full-width cross-block joint penalties (the reference-symmetric
    // `M⊗S_t` multinomial smoothing penalty). Empty for every other family, so
    // the joint-penalty code paths below are skipped and behaviour is identical.
    let reduced_total: usize = specs.iter().map(|s| s.design.ncols()).sum();
    let joint_specs = pulled_back_joint_penalty_specs(family, &canonical, reduced_total)?;

    let label_layout = penalty_label_layout_with_joint(specs, penalty_counts.clone(), joint_specs)?;
    let mut rho0 = label_layout.initial_rho.clone();
    // One warm-start source per fit: a `warm_start_from` point (gam#3002) replaces
    // the persistent store's records and artifacts.
    let (persistent_warm_start_cache, mut persistent_warm_start) = if options.warm_start.is_some() {
        (None, None)
    } else {
        load_persistent_custom_family_warm_start::<F>(family, specs, options, rho0.len())
    };
    // The cross-fit `FitArtifact` transfer (consume/capture below) reuses
    // per-block β/ρ from a structurally-matching prior fit under a descriptor
    // key that deliberately EXCLUDES the response. Per the
    // `persistent_warm_start_fingerprint` contract, reusing β across fits is
    // only admissible for families that opt into persistent warm-starts by
    // providing a likelihood-data fingerprint (which is exactly what makes
    // `persistent_warm_start_cache` `Some`). Families that opt out (fingerprint
    // `None` ⇒ key `None`) must cold-start so repeat fits of the same model are
    // bit-reproducible: without this gate a second structurally-identical fit
    // warm-starts off the first and settles on a different point within the
    // inner solve's flat-basin tolerance (gam#1607 cluster 4 — the location-
    // scale engine-vs-reference exact-replay parity), and successive process
    // runs drift as each seeds off the previous run's on-disk artifact.
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
    if persistent_warm_start.is_none() && !rho0.is_empty() {
        if let Some(cache) = persistent_warm_start_cache.as_ref() {
            if let Some(warm) = consume_fit_artifact::<F>(
                &cache.store,
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
        .map_err(|mut refusal| {
            refusal.map_descending_ray_direction(&|reduced| {
                lift_direction_to_raw(&canonical.gauge, reduced)
            });
            refusal
        })?;
        let warm_start = constrained_warm_start_from_inner(&rho0, &inner);
        // An unconverged solve never seeds a later fit (#2902).
        if inner.converged {
            store_persistent_custom_family_warm_start(
                persistent_warm_start_cache.as_ref(),
                specs,
                &warm_start,
            );
        }
        if !inner.converged {
            // The terminal verdict travels typed, as the fixed-log-lambda route
            // returns it (#1561): the Jeffreys arming lifecycle reads its terminal
            // reason as arming evidence, and text carries none (#979).
            let mut refusal = inner_solve_not_converged_error(&inner, options, 0, 0);
            refusal.map_descending_ray_direction(&|reduced| {
                lift_direction_to_raw(&canonical.gauge, reduced)
            });
            return Err(refusal);
        }
        refresh_all_block_etas(family, specs, &mut inner.block_states)?;
        audit_converged_identifiability(family, raw_specs, &canonical, &inner.block_states, 0)?;
        let hessian = materialize_owned_terminal_unpenalized_hessian(
            family,
            specs,
            &inner.block_states,
            inner.joint_workspace.as_ref(),
            inner.terminal_working_sets.as_deref(),
            "custom-family no-smoothing terminal Hessian",
        )
        .map_err(|reason| CustomFamilyError::Optimization {
            context: "fit_custom_family no-smoothing terminal curvature ownership",
            reason: reason.to_string(),
        })?;
        let posterior = compute_joint_posterior(
            family,
            specs,
            &inner.block_states,
            &per_block,
            options,
            Some(&hessian),
            inner.terminal_working_sets.as_deref(),
            inner.joint_workspace.as_ref(),
            inner.terminal_likelihood_score.as_ref(),
        )
        .map_err(|reason| CustomFamilyError::Optimization {
            context: "fit_custom_family no-smoothing terminal posterior",
            reason: format!("{reason}; no fit was assembled"),
        })?;
        let JointPosteriorAssembly {
            covariance_conditional,
            mut geometry,
            reported_beta,
            improper_penalty_null_posterior,
        } = posterior;
        let reml_term = if options.use_remlobjective {
            let logdet_h = inner
                .block_logdet_h
                .ok_or_else(|| CustomFamilyError::Optimization {
                    context: "fit_custom_family no-smoothing inner solve",
                    reason: "certified inner mode is missing its Hessian logdet".to_string(),
                })?;
            let logdet_s = inner
                .block_logdet_s
                .ok_or_else(|| CustomFamilyError::Optimization {
                    context: "fit_custom_family no-smoothing inner solve",
                    reason: "certified inner mode is missing its penalty logdet".to_string(),
                })?;
            0.5 * (logdet_h - logdet_s)
        } else {
            0.0
        };
        let penalized_objective = checked_penalizedobjective(
            inner.log_likelihood,
            inner.penalty_value,
            reml_term,
            "custom-family fit without smoothing parameters",
        )
        .map_err(|reason| CustomFamilyError::Optimization {
            context: "fit_custom_family no-smoothing penalized objective",
            reason: reason.to_string(),
        })?;
        // Cross-fit FitArtifact capture (Phase 0/1): persist the converged
        // raw-β + ρ under the descriptor-indexed keyspace so a later fold
        // can warm-start its ρ. Best-effort; never affects this fit. Gated on
        // the same opt-in as the consume side (gam#1607) so opt-out families
        // publish nothing and stay bit-reproducible across repeat fits/runs.
        if let Some(cache) = persistent_warm_start_cache.as_ref() {
            capture_fit_artifact::<F>(
                &cache.store,
                specs,
                &canonical.gauge,
                &warm_start.block_beta,
                &warm_start.rho,
                &label_layout.physical_to_outer,
                penalized_objective,
                true,
            );
        }
        install_reported_posterior_mean(
            family,
            specs,
            &mut inner,
            &mut geometry,
            reported_beta.as_ref(),
        )
        .map_err(|reason| CustomFamilyError::Optimization {
            context: "fit_custom_family no-smoothing reported posterior mean",
            reason: reason.to_string(),
        })?;
        let geometry = Some(geometry);
        let deviance = classical_deviance_at_mode(
            family,
            &inner.block_states,
            "fit_custom_family no-smoothing classical deviance",
        )?;
        return assemble_custom_family_fit_result(
            inner,
            BlockwiseFitAssembly {
                rho_physical: physical_rho0,
                deviance,
                covariance_conditional,
                geometry,
                precomputed_edf: None,
                canonical: Some(&canonical),
                result_specs: raw_specs,
                penalized_objective,
                outer_iterations: 0,
                outer_gradient_norm: None,
                criterion_certificate: None,
                outer_converged: true,
                joint_log_lambdas: None,
                smoothing_corrected: None,
                smoothing_correction_absence: None,
                coefficient_mode_selection: if family.exact_newton_joint_hessian_beta_dependent()
                    && !family.inner_coefficient_objective_is_globally_convex()
                {
                    gam_solve::model_types::CoefficientModeSelection::SeedSelected {
                        reason: "the fit has no smoothing parameter to anchor at".to_string(),
                    }
                } else {
                    gam_solve::model_types::CoefficientModeSelection::UniqueMode
                },
                improper_penalty_null_posterior,
            },
        );
    }

    use gam_problem::OuterEval;
    use gam_solve::model_types::EstimationError;
    use gam_solve::rho_optimizer::{OuterEvalOrder, OuterProblem};

    // #2349 — shared "re-evaluate COLD" pulse. The outer cost-stall guard raises
    // it when it grants a STUCK-stall escape (a near-separating profiled fit
    // whose warm-started trajectory carries value hysteresis on a near-flat
    // inner ridge). The outer-eval closures below observe it, drop the warm
    // cache, and re-solve the inner problem cold so search descends a consistent
    // objective surface instead of grinding to `max_iter` at a non-stationary
    // point.
    let outer_force_cold = Arc::new(AtomicBool::new(false));
    // #2668 — accepted outer steps, advanced by the optimizer's accept observer.
    // The outer-eval closures read it so that only an accepted iterate's inner
    // mode seeds the search, and a rejected trial never does.
    let outer_accepted_steps = Arc::new(AtomicUsize::new(0));
    let outer_options = options.clone();

    let n_rho = rho0.len();
    let (cap_gradient, cap_hessian) =
        custom_family_outer_derivatives(family, specs, &outer_options);
    let outer_hessian_absence =
        crate::joint_newton::custom_family_outer_hessian_absence(family, specs, &outer_options);
    let derivative_policy = family.outer_derivative_policy(specs, &outer_options);
    let hessian = cap_hessian;
    let need_outer_hessian = hessian.is_analytic();
    log::debug!(
        "[OUTER] custom family derivative-policy: n_params={} gradient={:?} hessian={:?} capability={:?} requested_outer_hessian={} inner_hvp_available={} outer_hvp_available={} outer_dense_available={}",
        n_rho,
        cap_gradient,
        hessian,
        derivative_policy.capability,
        need_outer_hessian,
        family.inner_coefficient_hessian_hvp_available(specs),
        family.outer_hyper_hessian_hvp_available(specs),
        family.outer_hyper_hessian_dense_available(specs),
    );
    let bfgs_step_cap = Some(FIRST_ORDER_BFGS_LOGLAMBDA_STEP_CAP);
    // EFS / HybridEfs structural property (`H^{-1/2} B_k H^{-1/2} ≽ 0` plus a
    // parameter-independent nullspace, Wood-Fasiolo) fails for multi-block
    // families whose joint likelihood Hessian depends on β.
    let multi_block_beta_dependent =
        specs.len() > 1 && family.exact_newton_joint_hessian_beta_dependent();
    // The criterion of a constrained family prices the cone normalizer, which the EFS fixed
    // point does not model (gam#2765).
    let prices_cone_normalizer = pre_fit_declares_linear_constraints(family, raw_specs);
    // Calibrate the outer solver to the n-scaled profiled REML/LAML objective.
    // The profiled criterion is a sum over n observations, so |f| ~ O(n) for
    // every family. Without this calibration the outer search uses a bare
    // absolute gradient floor of `outer_tol ≈ 1e-5`, forcing iteration until
    // |g| ≤ 1e-5 even when |f| ~ 200 and τ·(1+|f|) ~ 2e-3 already signals
    // convergence in the relative-to-cost sense.
    // Mirroring the spatial exact-joint outer fix (#1053/#1066/#1069) and
    // the primary REML outer (solver/estimate.rs) for the custom-family path.
    let n_obs = specs.first().map(|s| s.design.nrows()).unwrap_or(0);
    let p_total: usize = specs.iter().map(|s| s.design.ncols()).sum();
    // Establish the ρ box once, validated, so its floor and ceiling reach both
    // the optimizer bounds and the per-term tightening from a single source
    // that cannot be transposed (#2370).
    // The per-coordinate ρ at which each penalized term's structural effective
    // df reaches one. This is both the optimizer's upper bound and — because
    // every term sits on its penalty nullspace there, leaving a unique
    // parametric mode — the anchor of the #2366 continuation below.
    let ResolvabilityRhoDomain {
        lower: rho_lower_bounds,
        upper: rho_upper_bounds,
        lower_is_limit: rho_lower_is_limit,
        upper_is_limit: rho_upper_is_limit,
    } = resolvability_rho_domain_and_limit_faces(
        specs,
        &label_layout,
        n_rho,
        options.rho_lower_bound,
    )?;
    // The #2366 continuation anchor is the MAXIMAL-smoothing ρ — the uniform
    // ceiling — and not the per-term upper bound above.
    //
    // The two coincided while the effective-df floor was absolute: a term that
    // could not reach df = 1 was exempt, so its bound WAS the ceiling. #2608
    // made the floor relative for exactly those terms, holding a rank-deficient
    // term at a fraction of the df it can reach. That is a deliberately
    // PARTIAL collapse, which is the opposite of what the anchor needs: the
    // anchor's whole justification is that every penalized term sits on its
    // penalty nullspace there, leaving a unique parametric mode that the
    // caller's coefficients cannot select. Retaining half of a rank-1 term's
    // attainable df leaves it at ρ = ln γ, where a nonconvex inner problem
    // still has all of its modes — anchoring there hands branch selection back
    // to the seed, which is the property #2366 exists to remove.
    //
    // The anchor is therefore derived from the ρ-box's own ceiling. It is not a
    // fresh constant, and it can only ever sit at or above the per-term bound,
    // so the continuation still starts at the most-smoothed admissible point.
    let continuation_anchor = rho_upper_bounds.clone();

    // #2366: for a family whose complete coefficient objective is nonconvex,
    // `argmin_θ ℓ_p(θ, ρ)` can contain disconnected modes and the outer
    // criterion is only a function of ρ once a selection rule is fixed. Seed
    // the search with the mode that rule names — the endpoint of the
    // continuation from the maximal-smoothing anchor — instead of with whatever
    // mode the caller's coefficients happen to reach.
    //
    // Hessian β-dependence remains necessary because a β-independent Hessian
    // defines a quadratic objective, but it is not sufficient for nonconvexity:
    // multinomial logit's `XᵀW(β)X` varies with β while remaining PSD. A family
    // that certifies global convexity has no competing basins, so an
    // exponentially refined branch-tracking continuation is both mathematically
    // irrelevant and potentially much more expensive than the fit itself.
    // A family-owned objective homotopy is the strongest mode definition: its
    // zero member supplies a unique coefficient mode at the caller's actual
    // rho, and the path changes only the family augmentation being armed. In
    // particular, a conditional Firth refit must follow Jeffreys strength
    // 0→1 from the already-certified unbiased mode; walking rho from maximal
    // smoothing solves a different selection problem and can discard that
    // authoritative mode before the Firth objective is even reached.
    // A continuation that cannot certify its mode selection DECLINES; it does
    // not kill the fit (#2612).
    //
    // Its sibling `anchored_continuation_seed` has carried that contract since
    // #2366 — "the production caller logs a refusal and keeps its existing seed,
    // so declining a continuation still never turns a fit that works today into
    // a failure" — and this call site was the one place that read the same kind
    // of refusal as fatal. The asymmetry does not survive stating: refusing the
    // whole fit does not make `V(ρ)` well defined, it only denies the caller the
    // answer that the pre-#2366 seed would have produced. Measured on the
    // penguins stride-4 armed refit, that asymmetry was the entire user-visible
    // failure — a `FIT FAILED after 8.9s` where the seed the caller had already
    // certified was sitting in `persistent_warm_start`.
    //
    // What a decline costs is real and is logged rather than hidden: the mode is
    // then selected by the caller's coefficients, so `θ̂` is a functional of the
    // seed for this fit and the #2366 guarantee does not hold for it.
    let objective_homotopy_seed = match coefficient_objective_homotopy_seed(
        family,
        specs,
        &outer_options,
        &label_layout,
        &rho_prior,
        &rho0,
    ) {
        Ok(seed) => seed,
        Err(refusal) => {
            log::debug!(
                "[OUTER] coefficient-objective continuation declined with typed refusal: \
                 {refusal}. The armed coefficient mode is therefore selected by the caller's \
                 seed rather than by the continuation, so it is not the #2366 canonical mode \
                 for this fit"
            );
            None
        }
    };
    // The rule that selected the mode is recorded on the fit (#2661), so a
    // declined continuation is visible to a caller, not only in this log.
    let mode_seed = if let Some(certified) = objective_homotopy_seed {
        log::debug!(
            "[OUTER] coefficient-objective continuation certified at {} steps: endpoint \
             discrepancy {:.3e} <= inner tolerance {:.3e}; observed contraction factor {:?}",
            certified.certificate.steps,
            certified.certificate.endpoint_discrepancy,
            certified.certificate.inner_tolerance,
            certified.certificate.observed_contraction_factor,
        );
        (
            Some(certified.warm_start),
            gam_solve::model_types::CoefficientModeSelection::ObjectiveHomotopy {
                steps: certified.certificate.steps,
            },
        )
    } else if inner_objective_may_have_several_modes(family) {
        match anchored_continuation_seed(
            family,
            specs,
            &outer_options,
            &label_layout,
            &rho_prior,
            &continuation_anchor,
            &rho0,
        ) {
            Ok(certified) => {
                log::debug!(
                    "[OUTER] #2661 anchored continuation certified at {} steps: endpoint \
                     discrepancy {:.3e} <= inner tolerance {:.3e}; observed contraction \
                     factor {:?}",
                    certified.certificate.steps,
                    certified.certificate.endpoint_discrepancy,
                    certified.certificate.inner_tolerance,
                    certified.certificate.observed_contraction_factor,
                );
                (
                    Some(certified.warm_start),
                    gam_solve::model_types::CoefficientModeSelection::AnchoredContinuation {
                        steps: certified.certificate.steps,
                        endpoint_discrepancy: certified.certificate.endpoint_discrepancy,
                    },
                )
            }
            Err(refusal) => {
                log::debug!(
                    "[OUTER] #2661 anchored continuation declined with typed refusal: {refusal}"
                );
                (
                    persistent_warm_start.clone(),
                    declined_continuation_selection(&refusal),
                )
            }
        }
    } else {
        (
            persistent_warm_start.clone(),
            gam_solve::model_types::CoefficientModeSelection::UniqueMode,
        )
    };
    let (initial_warm_cache, coefficient_mode_selection) = mode_seed;
    // What "cold" means when the stall guard drops the warm cache. Dropping it
    // to `None` sends the inner solve back to whatever coefficients the caller
    // supplied — trajectory-independent, but arbitrary, and for a nonconvex
    // family that is a seed selecting a branch. The point of the pulse is to
    // re-solve on a surface that does not depend on the path taken, so the
    // fallback is the anchored mode: cold means CANONICAL, not arbitrary.
    let canonical_seed = initial_warm_cache.clone();
    // gam#3173: the fit's fixed starts, the anchored mode and the caller's seed. Every evaluation
    // also solves a mode from each and publishes the certified one with the lowest penalized
    // objective (`evaluate_on_branch`). A family whose inner objective has one mode needs none.
    let fixed_starts = fixed_mode_starts(
        family,
        [canonical_seed.clone(), persistent_warm_start.clone()],
    );
    let problem = OuterProblem::new(n_rho)
        .with_stuck_stall_cold_reeval_signal(
            Arc::clone(&outer_force_cold),
            Arc::clone(&outer_accepted_steps),
        )
        .with_gradient(cap_gradient)
        .with_hessian(hessian)
        // #2359's optimize-3/certify-4 lifecycle (#2898), for an armed Jeffreys
        // term only. The exact Hessian stays declared, and the terminal mint
        // requests `ValueGradientHessian` from that declaration whatever the
        // search plan is. With the term armed, the search runs BFGS on the exact
        // analytic gradient, so the order-five Jeffreys curvature (D²H_Φ, the
        // completion pair correction, the third information derivative) is
        // priced once at the certificate instead of on every ARC trial, rejected
        // trials included. At 2f844874e on survival marginal-slope 160×6, ARC
        // search took 178.2 s to V=264.68231024 with 11 strict-saddle windows,
        // and order five was 59-60% of that time; gradient-only search took
        // 19.6 s to V=264.68203561 and minted (6,0,0) with λ_min=1.68e-4.
        //
        // An unarmed family has no order-five pieces, so that saving does not
        // exist, and the exact-curvature search is the cheaper plan (#3306). On
        // the unarmed binary Bernoulli marginal-slope fit (80,016 rows, p=81,
        // 13 ρ), BFGS spent ~380 s per seed and ended in a line-search
        // refusal, while ARC reached the certified value in 16-19 evaluations
        // (~40-95 s).
        .with_prefer_gradient_only(family.joint_jeffreys_term_required())
        // The mode-selection consumer below requires a certified local minimum,
        // not merely a stationary point whose raw negative curvature was cleared
        // by the generic gradient-residue floor. Declare that requirement before
        // optimization so the mint can escape and re-optimize such a saddle.
        //
        // Gated on `need_outer_hessian` (this family's own
        // `DeclaredHessianForm::is_analytic()`, computed above) because the
        // requirement is only satisfiable when the family exposes an analytic
        // outer Hessian. `custom_family_outer_derivatives` returns
        // `DeclaredHessianForm::Unavailable` unless `use_outer_hessian`,
        // `include_exact_newton_logdet_h` and `policy.capability.has_hessian()`
        // all hold; the middle one needs `exact_newton_outerobjective()` to be
        // `RidgedQuadraticReml` or `StrictPseudoLaplace`, which the dispersion
        // location-scale families are not.
        //
        // When the form is `Unavailable`, `certify_fixed_point_optimality`
        // never requests curvature (`wants_analytic_hessian` is false), so
        // `certificate.hessian_psd()` is `None` — and
        // `certificate_meets_curvature_requirement` treats `None` exactly like
        // `Some(false)`. The fit is then refused for an ABSENCE of evidence
        // rather than evidence of a saddle, with the escape path this
        // requirement was declared to enable unable to engage because there is
        // no measured curvature to escape from. Measured on a converged fit:
        //
        //   |Pg|=4.008e-3  bound=8.464e-3  -> stationary
        //   hessian_psd=n/a  curvature_source=unavailable
        //   plan=solver=Efs  claimed_converged=true  |step|=0.000000e0
        //
        // Demanding a measurement the family cannot produce refuses every such
        // fit unconditionally. Where the family DOES expose an analytic Hessian
        // the requirement is unchanged and still binds, saddle escape included.
        .with_require_measured_psd(need_outer_hessian)
        .with_disable_fixed_point(multi_block_beta_dependent || prices_cone_normalizer)
        .with_tolerance(options.outer_tol)
        .with_max_iter(options.outer_max_iter)
        .with_bfgs_step_cap(bfgs_step_cap)
        .with_initial_rho(rho0.clone())
        .with_problem_size(n_obs, p_total.max(1))
        // Per-coordinate ρ domain (#2812): the interval on which each term's
        // penalty is resolvable against its own design curvature, derived in
        // `resolvability_rho_domain`. Its lower edge is where the term is
        // unpenalized to the gradient's resolution, its upper edge where the term sits
        // on its penalty null space to the gradient's resolution; a search that
        // reaches either has found a structural result, not a wall. The hand
        // ceiling of 12 and the effective-df floor of one that used to bound
        // this search are gone with it.
        .with_bounds(rho_lower_bounds.clone(), rho_upper_bounds.clone())
        // #2954: which of those edges are the terms' limit models, so the mint may
        // rail an exponential tail there and refuses by type at a literal face.
        .with_limit_faces(rho_lower_is_limit, rho_upper_is_limit);
    // The search enters from the one start `rho0`: the caller's pinned ρ, the
    // validated warm ρ, or the derived cold start. No seed lattice is ranked
    // in front of it; the certified outer search owns every move from there.
    let warm_start_present = persistent_warm_start.is_some();
    // A low-level caller-keyed session wins. Otherwise derive the outer stream
    // from the same explicit store and structural key used by the block record
    // and cross-fit artifact owners.
    // One warm-start source per fit: a warm start replaces the opportunistic
    // outer cache.
    let cache_session = if options.warm_start.is_some() {
        None
    } else {
        options.cache_session.clone().or_else(|| {
            persistent_warm_start_cache.as_ref().and_then(|cache| {
                gam_solve::persistent_warm_start::open_outer_session(&cache.store, &cache.key)
            })
        })
    };
    let outer_cache_attached = cache_session.is_some();
    let problem = if let Some(session) = cache_session {
        let key_hex = session.key().to_hex();
        log::debug!(
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
    // The one warm-start rule (gam#3002, `OuterProblem::with_warm_start`): on the
    // parent's inputs the point resumes the search that certified it, and every
    // other search runs cold; on other inputs it can only join the independent
    // multistart. A point of another width belongs to another search.
    let problem = match options.warm_start.as_ref() {
        Some(warm_start) => problem.with_warm_start(warm_start),
        None => problem,
    };

    // An inner failure at one trial rho makes that trial infeasible, not the
    // entire smoothing problem. Let the outer optimizer retreat and try its
    // remaining certified strategies. If none reaches stationarity, the outer
    // result below is returned as nonconvergence with checkpoint evidence;
    // no fit or posterior approximation is assembled from the trial state.
    let eval_outer = |family: &F,
                      outer: &mut CustomOuterState,
                      rho: &Array1<f64>,
                      order: OuterEvalOrder|
     -> Result<OuterEval, EstimationError> {
        // #2349: consume the cold-reeval pulse once per outer evaluation. When
        // active (a near-separating warm-start-hysteresis stall the outer guard
        // flagged), every inner solve in this evaluation drops the warm cache
        // and runs cold so search descends a trajectory-independent objective
        // surface.
        let force_cold = outer.take_force_cold();
        // #2668: a step the optimizer accepted since the previous evaluation
        // promotes its iterate's mode to the seed before this evaluation reads it.
        outer.adopt_accepted_steps();
        // A failed evaluation prices no criterion, so it publishes no rank (#2765).
        outer.last_criterion_rank = None;
        // Genuinely value-only fulfilment (#979). A `Value` request from an outer
        // cost or reactive-domain probe never consumes the outer
        // gradient. The inner solve in `EvalMode::ValueOnly` already produces the
        // converged block β; surface it as `inner_beta_hint` with a zero-length
        // gradient and skip the full k²·n·p² coupled-joint LAML gradient assembly.
        // A value probe seeds nothing: only an accepted iterate's mode does (#2668).
        if matches!(order, OuterEvalOrder::Value) {
            let seed_identity = crate::warm_start::SeedIdentity::of(outer.seed_for(rho));
            let starts = if force_cold {
                ModeStarts {
                    incumbent: canonical_seed.as_ref(),
                    fixed: &fixed_starts,
                }
            } else {
                outer.mode_starts_for(rho)
            };
            return match evaluate_on_branch(
                family,
                specs,
                &outer_options,
                &label_layout,
                rho,
                starts,
                &rho_prior,
                EvalMode::ValueOnly,
            ) {
                Ok(eval) if eval.inner_converged && eval.objective.is_finite() => {
                    crate::warm_start::publish_outer_selected_evaluation(&eval);
                    // The gradient at this θ starts from the same seed and would
                    // re-derive this mode; it is served there instead (#979).
                    if !force_cold {
                        outer.record_value_probe(rho, seed_identity, eval.warm_start.clone());
                    }
                    outer.last_criterion_rank = eval.criterion_rank;
                    let inner_beta_hint = Some(Array1::from_iter(
                        eval.warm_start
                            .block_beta
                            .iter()
                            .flat_map(|beta| beta.iter().copied()),
                    ));
                    outer.last_error = None;
                    Ok(OuterEval {
                        cost: eval.objective,
                        gradient: Array1::zeros(rho.len()),
                        hessian: gam_problem::HessianValue::Unavailable,
                        inner_beta_hint,
                    })
                }
                Ok(eval) => {
                    let failure = if eval.inner_converged {
                        CustomFamilyError::trial_point(format!(
                            "custom-family value-only outer objective was non-finite ({})",
                            eval.objective
                        ))
                    } else {
                        inner_solve_not_converged_error(&eval.inner, &outer_options, rho.len(), 0)
                    };
                    // A value probe never seeds the next evaluation (#2668), and an
                    // unconverged solve never does either (#2902).
                    outer.record_refusal(failure);
                    Ok(OuterEval::infeasible(rho.len()))
                }
                Err(e) => {
                    // The outer-objective evaluator's whole contract is "produce
                    // the REML/LAML objective AT THIS rho", so every failure it
                    // can report is a failure to evaluate at this rho, and the
                    // outer optimizer's documented response to that is to reject
                    // the trial and step away. Stringifying it into
                    // `RemlOptimizationFailed` erased that: the variant carries
                    // prose, `is_trial_point_infeasible` answers `false` for it,
                    // and `into_objective_error` therefore graded a rho-local
                    // refusal `Fatal`. A correctly typed
                    // `CustomFamilyError::InnerSolveNotConverged` — a variant
                    // that says in its own name that the trial point is
                    // infeasible — died the same death here, which is why fixing
                    // only the producer's type changed the message and not the
                    // outcome (#2590).
                    //
                    // The classification is not a guess. If the cause really is
                    // rho-independent it recurs at every probed rho, the seed
                    // loop exhausts, and the run still ends — with this same
                    // reason quoted, after a bounded number of cheap identical
                    // failures. The reverse mistake killed fits that were
                    // perfectly fittable one rho away (#2553, #2590). The two
                    // costs are not comparable, so the boundary takes the safe
                    // one.
                    //
                    // `into_trial_point` and not `trial_point`: the evaluator
                    // now hands back a typed `CustomFamilyError`, and one that
                    // already answers this question must be passed through, not
                    // re-wrapped -- re-wrapping rendered it to text and prefixed
                    // it a second time (gam#2667).
                    let failure = e.into_trial_point();
                    outer.record_refusal(failure.clone());
                    Err(EstimationError::CustomFamily(failure))
                }
            };
        }
        let request_hessian =
            matches!(order, OuterEvalOrder::ValueGradientHessian) && need_outer_hessian;
        // Only a successful derivative-bearing evaluation may own the mode
        // consumed by certified fit assembly. A failed analytic probe must not
        // leave an older mode available for accidental substitution.
        outer.begin_terminal_evaluation();
        let starts = if force_cold {
            ModeStarts {
                incumbent: canonical_seed.as_ref(),
                fixed: &fixed_starts,
            }
        } else {
            outer.mode_starts_for(rho)
        };
        let eval_result = match evaluate_on_branch(
            family,
            specs,
            &outer_options,
            &label_layout,
            rho,
            starts,
            &rho_prior,
            if request_hessian {
                EvalMode::ValueGradientHessian
            } else {
                EvalMode::ValueAndGradient
            },
        ) {
            Ok(eval) if !eval.inner_converged => {
                let failure = inner_solve_not_converged_error(&eval.inner, &outer_options, rho.len(), 0);
                // An unconverged solve never seeds the next evaluation (#2902).
                outer.record_refusal(failure);
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
                outer.record_first_order_mode(warm_start.clone());
                store_persistent_custom_family_warm_start(
                    persistent_warm_start_cache.as_ref(),
                    specs,
                    &warm_start,
                );
                outer.last_error = None;
                outer.last_criterion_rank = eval.criterion_rank;
                eval
            }
            Ok(eval) => {
                // Name the failing channel. A requested Hessian that is simply unavailable
                // was reported as "non-finite", which is how #979's terminal refusal came to
                // read "value or gradient is non-finite" at a state whose search evaluations
                // were all finite (job 1131465).
                let channel = if !eval.objective.is_finite() {
                    "the objective is non-finite"
                } else if !eval.gradient.iter().all(|v| v.is_finite()) {
                    "the outer gradient is non-finite"
                } else if matches!(eval.outer_hessian, gam_problem::HessianValue::Unavailable) {
                    "the outer Hessian was requested and is unavailable"
                } else {
                    "the outer Hessian is non-finite or does not match the outer dimension"
                };
                outer.record_refusal(CustomFamilyError::trial_point(format!(
                    "custom-family outer evaluation refused: {channel} (objective={})",
                    eval.objective
                )));
                // Recoverable (data-driven): the objective/derivatives became
                // non-finite at this trial rho (e.g. separation / near-singular
                // information), so the outer optimizer retreats from this infeasible
                // point rather than the whole run hard-erroring. Exhausting
                // those alternatives becomes a terminal nonconvergence error.
                return Ok(OuterEval::infeasible(rho.len()));
            }
            Err(e) => {
                // A failure to evaluate the objective at this rho is a
                // statement about this rho — see the value-only probe above for
                // why the boundary owns that classification (#2590), and why a
                // typed error that already says so is passed through (#2667).
                let failure = e.into_trial_point();
                outer.record_refusal(failure.clone());
                return Err(EstimationError::CustomFamily(failure));
            }
        };
        crate::warm_start::publish_outer_selected_evaluation(&eval_result);
        let inner_beta_hint = Some(Array1::from_iter(
            eval_result
                .warm_start
                .block_beta
                .iter()
                .flat_map(|beta| beta.iter().copied()),
        ));
        let objective = eval_result.objective;
        let gradient = eval_result.gradient;
        let outer_hessian = eval_result.outer_hessian;
        let mode = CustomFamilyOwnedMode {
            objective,
            // `pullback_labeled_outer_eval` has already made `rho` the
            // semantic outer coordinate and added the labeled prior. Retain
            // that coordinate rather than the evaluator's expanded physical
            // smoothing vector.
            rho: rho.clone(),
            hyper_values: Array1::zeros(0),
            inner: eval_result.inner,
            psi_scores: None,
        };
        log::trace!(
            "[OUTER-EVAL] order={order:?} request_hessian={request_hessian} cost={objective:.6e} \
             |g|={:.6e} warm={} rho0={:.4}",
            gradient.iter().map(|g| g * g).sum::<f64>().sqrt(),
            outer.warm_cache.is_some(),
            rho[0],
        );
        outer.install_terminal_mode(rho, objective, &gradient, mode);
        Ok(OuterEval {
            cost: objective,
            gradient,
            hessian: outer_hessian,
            inner_beta_hint,
        })
    };

    // One complete outer search from `problem`'s seeds, with its own objective
    // state, stall pulse and accepted-step counter (#2349, #2668). A parallel
    // multistart runs one per seed (gnomon#2359).
    let run_outer =
        |family: &F,
         problem: &OuterProblem,
         force_cold: Arc<AtomicBool>,
         accepted_steps: Arc<AtomicUsize>|
         -> (
            Result<gam_solve::rho_optimizer::CertifiedOuterResult, EstimationError>,
            CustomOuterState,
        ) {
            let mut obj = problem.build_objective_with_eval_order(
        CustomOuterState::new_with_cold_signal(
            initial_warm_cache.clone(),
            force_cold,
            accepted_steps,
        )
        .with_outer_derivative_pilot(family.outer_derivative_pilot_schedule())
        .with_fixed_starts(fixed_starts.clone()),
        |outer: &mut CustomOuterState, rho: &Array1<f64>| {
            // Start from the incumbent's inner mode when there is one — a converged
            // inner solution gives a much better starting point. This was previously
            // disabled for exact-Hessian families, forcing every inner solve to start
            // from scratch (5-10 Newton steps instead of 1-2 with warm start).
            //
            // #2349: once the outer cost-stall guard has raised the cold-reeval
            // pulse (near-separating warm-start hysteresis), drop the warm cache
            // and run this probe cold so the profiled objective is a consistent
            // function of ρ.
            let force_cold = outer.take_force_cold();
            // #2668: an accepted step promotes its iterate's mode before this probe
            // reads the seed, and the probe itself never replaces it.
            outer.adopt_accepted_steps();
            outer.last_criterion_rank = None;
            let seed_identity = crate::warm_start::SeedIdentity::of(outer.seed_for(rho));
            let starts = if force_cold {
                ModeStarts {
                    incumbent: canonical_seed.as_ref(),
                    fixed: &fixed_starts,
                }
            } else {
                outer.mode_starts_for(rho)
            };
            match evaluate_on_branch(
                family,
                specs,
                &outer_options,
                &label_layout,
                rho,
                starts,
                &rho_prior,
                EvalMode::ValueOnly,
            ) {
                Ok(eval) if eval.inner_converged && eval.objective.is_finite() => {
                    crate::warm_start::publish_outer_selected_evaluation(&eval);
                    // The gradient at this θ starts from the same seed and would
                    // re-derive this mode; it is served there instead (#979).
                    if !force_cold {
                        outer.record_value_probe(rho, seed_identity, eval.warm_start.clone());
                    }
                    outer.last_criterion_rank = eval.criterion_rank;
                    outer.last_error = None;
                    Ok(eval.objective)
                }
                Ok(eval) => {
                    let failure = if eval.inner_converged {
                        CustomFamilyError::trial_point(format!(
                            "custom-family value-only outer objective was non-finite ({})",
                            eval.objective
                        ))
                    } else {
                        inner_solve_not_converged_error(&eval.inner, &outer_options, rho.len(), 0)
                    };
                    // A value probe never seeds the next evaluation (#2668), and an
                    // unconverged solve never does either (#2902).
                    outer.record_refusal(failure);
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
                    // A failure to evaluate the cost at this rho is a
                    // statement about this rho (#2590); a typed error that
                    // already says so is passed through, not re-prefixed
                    // (#2667).
                    let failure = e.into_trial_point();
                    outer.record_refusal(failure.clone());
                    Err(EstimationError::CustomFamily(failure))
                }
            }
        },
        |outer: &mut CustomOuterState, rho: &Array1<f64>| {
            eval_outer(
                family,
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
            eval_outer(family, outer, rho, order)
        },
        Some(|outer: &mut CustomOuterState| {
            outer.reset();
        }),
        Some(|outer: &mut CustomOuterState, rho: &Array1<f64>| {
            if !label_layout.supports_direct_physical_efs() {
                let failure = CustomFamilyError::UnsupportedConfiguration {
                    reason: "custom-family EFS requires an identity per-block \
                         penalty-coordinate layout with no fixed, tied, or joint penalties"
                        .to_string(),
                };
                outer.record_refusal(failure.clone());
                return Err(EstimationError::CustomFamily(failure));
            }
            let warm_ref = outer.warm_start_for(rho);
            match outerobjectiveefs(
                family,
                specs,
                &outer_options,
                &label_layout.penalty_counts,
                rho,
                warm_ref,
                rho_prior.clone(),
            ) {
                Ok((eval, warm, true, _inner)) => {
                    outer.warm_cache = Some(warm);
                    outer.last_error = None;
                    Ok(eval)
                }
                Ok((_eval, _warm, false, inner)) => {
                    let failure =
                        inner_solve_not_converged_error(&inner, &outer_options, rho.len(), 0);
                    // An unconverged solve never seeds the next evaluation (#2902).
                    outer.record_refusal(failure.clone());
                    // EFS cannot form a valid fixed-point update away from an
                    // inner mode. Returning an error lets the outer strategy
                    // runner try an analytically valid alternative; exhaustion
                    // remains a terminal nonconvergence error.
                    Err(EstimationError::CustomFamily(failure))
                }
                Err(e) => {
                    // A failure to build the EFS update at this rho is a
                    // statement about this rho (#2590).
                    let failure = e.into_trial_point();
                    outer.record_refusal(failure.clone());
                    Err(EstimationError::CustomFamily(failure))
                }
            }
        }),
    )
    .with_seed_inner_state(|outer: &mut CustomOuterState, beta: &Array1<f64>| {
        outer.seed_cached_beta(n_rho, specs, beta)
    })
    .with_exact_polish(CustomOuterState::begin_exact_polish)
    // #2765: the projected criterion prices `½·log|ZᵀMZ|₊` over a kept rank that moves
    // with the inner mode's face, so the outer search keeps each run on one rank.
    .with_criterion_rank(|outer: &CustomOuterState| outer.last_criterion_rank)
    // EFS may discover the optimum, but only the labeled analytic evaluator
    // owns the exact objective/gradient/coefficient-mode identity consumed by
    // fit assembly. Force the runner's final full-fidelity installation
    // through that evaluator regardless of the search plan.
    //
    // That installation needs objective, gradient and mode, not curvature
    // (#2898). The mint that follows at the same rho requests
    // `ValueGradientHessian` wherever the Hessian is declared, re-installs the
    // mode, and owns the Hessian the certificate judges and the smoothing
    // correction reads. Installing at order four as well assembled the outer
    // Hessian, order-five Jeffreys pieces included, twice at the selected point
    // and discarded the first.
    .with_terminal_eval_order(OuterEvalOrder::ValueAndGradient);
            let outer_result = problem.run_certified(&mut obj, "custom family");
            (outer_result, obj.state)
        };

    // A family that declares starts beside the fit's own searches each one in its
    // own run, in parallel, and keeps the best certified (gnomon#2359), when it runs
    // each search on its own member. A warm start or a cached outer session already
    // names the search's start, so those fits run the one search from it.
    let independent_search = family.independent_outer_search().filter(|search| {
        !search.additional_outer_start_levels().is_empty()
            && !warm_start_present
            && !outer_cache_attached
    });
    let (outer_result, mut outer_state) = if let Some(search) = independent_search {
        // The family predicts one search's working set with its caches at the
        // size they take running alone on the availability read here, once.
        let serial_available =
            gam_runtime::resource::resample_memory_availability().available_bytes();
        let working_set = search.outer_search_working_set_bytes(specs, serial_available);
        let multistart = problem.run_certified_multistart(
            &search.additional_outer_start_levels(),
            "custom family",
            serial_available,
            usize::try_from(working_set).unwrap_or(usize::MAX),
            |_, seed_problem, lane| {
                // Each seed searches on its own member of the family: its
                // per-fit state fresh and its memory choices read from its lane.
                let member = search.outer_search_member(lane);
                let force_cold = Arc::new(AtomicBool::new(false));
                let accepted_steps = Arc::new(AtomicUsize::new(0));
                let seed_problem = seed_problem.with_stuck_stall_cold_reeval_signal(
                    Arc::clone(&force_cold),
                    Arc::clone(&accepted_steps),
                );
                run_outer(&member, &seed_problem, force_cold, accepted_steps)
            },
        );
        match multistart {
            Ok(mut outcome) => match outcome.winner {
                Some(winner) => (outcome.outcomes.swap_remove(winner), outcome.payload),
                // No seed certified: refuse with every seed's outcome. The
                // first seed (the fit's own initial rho) lends its state for
                // the refusal's last-evaluation evidence.
                None => (Err(outcome.refusal("custom family")), outcome.payload),
            },
            Err(error) => (
                Err(error),
                CustomOuterState::new_with_cold_signal(
                    initial_warm_cache.clone(),
                    Arc::clone(&outer_force_cold),
                    Arc::clone(&outer_accepted_steps),
                ),
            ),
        }
    } else {
        run_outer(
            family,
            &problem,
            Arc::clone(&outer_force_cold),
            Arc::clone(&outer_accepted_steps),
        )
    };

    let last_error_detail = outer_state
        .last_error
        .as_ref()
        .map(|e| format!(" last objective error: {e}"))
        .unwrap_or_default();

    // SPEC 20: only the optimizer-owned certified carrier is fit authority.
    // Raw `OuterResult` status/certificate fields are intentionally
    // insufficient here; `run_certified` is the sole constructor that seals
    // the terminal analytic evidence after final-state installation.
    let certified_outer = match outer_result {
        Ok(outer) => outer,
        Err(e) => {
            // `rho_checkpoint` has ONE owner: `EstimationError::RemlDidNotConverge`
            // carries the best iterate and tells the caller to resume the outer
            // search there, and every other emission of the name in this file is
            // the ρ its fit is actually at (`rho_star`, or the caller's fixed ρ).
            // The warm cache is not that quantity — it is overwritten at EVERY
            // objective evaluation, including rejected trial steps, so on a
            // failed run it holds wherever the search died.
            // Emitting it under the same name minted a second definition, and the
            // two do diverge: on the #2501 by-group seed-3 refusal both appear in
            // ONE message, agreeing on seven of eight coordinates and differing by
            // a full 18.0 on the eighth, because the last evaluation was a
            // near-floor trial point while the best iterate sat interior.
            // `{e}` already carries the optimizer-owned checkpoint; this reports
            // the cache as what it is.
            let last_evaluated_rho = outer_state
                .warm_cache
                .as_ref()
                .map(|warm| warm.rho.to_vec())
                .unwrap_or_else(|| rho0.to_vec());
            if let Some(warm) = outer_state.warm_cache.as_ref() {
                store_persistent_custom_family_warm_start(
                    persistent_warm_start_cache.as_ref(),
                    specs,
                    warm,
                );
            }
            return Err(CustomFamilyError::OuterSmoothingFailed {
                reason: format!(
                    "outer smoothing optimization failed certified-fit validation after exhausting strategy fallbacks: \
                     {e}; last_evaluated_rho={last_evaluated_rho:?}; no fit was assembled.\
                     {last_error_detail}"
                ),
                last_refusal: outer_state.last_error.take().map(|mut refusal| {
                    refusal.map_descending_ray_direction(&|reduced| {
                        lift_direction_to_raw(&canonical.gauge, reduced)
                    });
                    Box::new(refusal)
                }),
                // The search's most recent uncertified inner solve, which a finite
                // trial after it does not clear. Only the fit boundary reads it
                // (#2943).
                search_inner_refusal: outer_state.last_inner_refusal.take().map(|mut refusal| {
                    refusal.map_descending_ray_direction(&|reduced| {
                        lift_direction_to_raw(&canonical.gauge, reduced)
                    });
                    Box::new(refusal)
                }),
                outer_error: std::sync::Arc::new(e),
            });
        }
    };
    // Consume the exact derivative-bearing evaluator state installed by the
    // runner's final full-fidelity synchronization. Objective, gradient,
    // smoothing coordinate, and coefficient mode form one sealed identity.
    // Warm starts are deliberately excluded: they are seeds, not evidence
    // about which nonconvex coefficient basin produced the certificate.
    let terminal = outer_state.terminal_mode.take().ok_or_else(|| {
        CustomFamilyError::Optimization {
            context: "fit_custom_family terminal mode ownership",
            reason: "outer optimization certified without retaining a derivative-bearing terminal coefficient mode; no fit was assembled"
                .to_string(),
        }
    })?;
    let rho_star = certified_outer.rho().clone();
    let mode = bind_certified_custom_family_terminal_mode(terminal, &certified_outer)?;
    let CustomFamilyOwnedMode {
        objective: penalized_objective,
        rho: mode_rho,
        hyper_values: mode_hyper_values,
        mut inner,
        psi_scores: _,
    } = mode;
    if !mode_hyper_values.is_empty() {
        return Err(CustomFamilyError::Optimization {
            context: "fit_custom_family terminal mode ownership",
            reason: "rho-only outer optimization retained unexpected non-rho coordinates"
                .to_string(),
        });
    }
    if !inner.converged {
        return Err(CustomFamilyError::Optimization {
            context: "fit_custom_family terminal mode ownership",
            reason:
                "the certified terminal coefficient mode was not converged; no fit was assembled"
                    .to_string(),
        });
    }
    let per_block = split_labeled_log_lambdas(&rho_star, &label_layout)?;
    let mut final_options = options.clone();
    // Reconstruct only the deterministic penalty geometry needed by covariance
    // and EDF assembly. The coefficient mode itself already came from this
    // exact rho-specific bundle and is never solved or evaluated again.
    if !label_layout.joint_specs.is_empty() {
        let total_compiled: usize = specs.iter().map(|s| s.design.ncols()).sum();
        let joint_log_lambdas = label_layout.joint_log_lambdas(&rho_star);
        let bundle = gam_problem::JointPenaltyBundle::from_validated_geometry(
            std::sync::Arc::clone(&label_layout.joint_specs),
            std::sync::Arc::clone(&label_layout.joint_roots),
            joint_log_lambdas,
            total_compiled,
        )
        .map_err(CustomFamilyError::from)?;
        final_options.joint_penalties = Some(std::sync::Arc::new(bundle));
    }

    let final_warm_start = constrained_warm_start_from_inner(&mode_rho, &inner);
    store_persistent_custom_family_warm_start(
        persistent_warm_start_cache.as_ref(),
        specs,
        &final_warm_start,
    );
    audit_converged_identifiability(
        family,
        raw_specs,
        &canonical,
        &inner.block_states,
        certified_outer.iterations(),
    )?;

    // Consume the exact returned-beta authority retained by the inner solve.
    // Coupled paths own a joint workspace; explicitly uncoupled paths own
    // terminal block working sets. Neither path re-evaluates the likelihood.
    let hessian = materialize_owned_terminal_unpenalized_hessian(
        family,
        specs,
        &inner.block_states,
        inner.joint_workspace.as_ref(),
        inner.terminal_working_sets.as_deref(),
        "custom-family certified terminal Hessian",
    )
    .map_err(|reason| CustomFamilyError::Optimization {
        context: "fit_custom_family terminal curvature ownership",
        reason: reason.to_string(),
    })?;
    let posterior = compute_joint_posterior(
        family,
        specs,
        &inner.block_states,
        &per_block,
        &final_options,
        Some(&hessian),
        inner.terminal_working_sets.as_deref(),
        inner.joint_workspace.as_ref(),
        inner.terminal_likelihood_score.as_ref(),
    )
    .map_err(|reason| CustomFamilyError::Optimization {
        context: "fit_custom_family final posterior assembly",
        reason: format!(
            "{reason}; rho_checkpoint={:?}; no fit was assembled",
            rho_star.as_slice().unwrap_or(&[])
        ),
    })?;
    let JointPosteriorAssembly {
        covariance_conditional,
        mut geometry,
        reported_beta,
        improper_penalty_null_posterior,
    } = posterior;
    // Cross-fit FitArtifact capture (Phase 0/1) for the converged smoothing
    // fit: persist the descriptor-indexed raw-β + ρ so a later fold transfers
    // ρ. Best-effort; never affects this fit's result. Gated on the same opt-in
    // as the consume side (gam#1607) so opt-out families publish nothing and
    // stay bit-reproducible across repeat fits/runs.
    if let Some(cache) = persistent_warm_start_cache.as_ref() {
        capture_fit_artifact::<F>(
            &cache.store,
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
    let physical_lambdas = exact_lambdas_from_log_strengths(
        &rho_star_physical,
        "custom-family terminal EDF log strength",
    )?;
    let precomputed_edf = if label_layout.joint_specs.is_empty() {
        Some(
            custom_family_blockwise_edf(
                geometry.penalized_hessian.as_array(),
                specs,
                &physical_lambdas.view(),
            )
            .map_err(|reason| CustomFamilyError::Optimization {
                context: "fit_custom_family terminal EDF",
                reason: reason.to_string(),
            })?,
        )
    } else {
        // The public per-penalty EDF vectors are aligned to per-block lambdas.
        // A full-width joint penalty has no truthful slot in that schema; do
        // not report p as if the owned joint penalty spent zero degrees of
        // freedom. `joint_log_lambdas` below preserves the selected strengths
        // until a typed joint-EDF channel exists.
        None
    };
    // gam#1587/#561: a family whose smoothing rides on the full-width JOINT
    // penalty (the multinomial centered `Σ_t λ_t (M ⊗ S_t)` metric) leaves its
    // per-block penalty lists — and hence the physical `rho_physical`/`lambdas`
    // expansion above — EMPTY, so the selected per-component `ρ_t` would be lost
    // at assembly. Surface it on `FitArtifacts.joint_log_lambdas` so the
    // reporting path can rebuild per-(class, term) λ and the influence-matrix
    // EDF. `None` (no allocation) for every per-block-only family.
    let joint_log_lambdas = (!label_layout.joint_specs.is_empty())
        .then(|| Array1::from(label_layout.joint_log_lambdas(&rho_star)));
    // #2346: first-order ρ-uncertainty smoothing correction for the joint
    // coefficient covariance, from the SAME analytic outer ρ-Hessian the
    // certificate judged. Rail coordinates (box rails + typed AsymptoteRail
    // rails) have no finite ρ-variance and are excluded (#2337 Thm 2.3);
    // directions under the certificate's gradient floor are dropped from V_ρ,
    // and a refused interior yields a typed absence, never an error.
    let (smoothing_corrected, smoothing_correction_absence) = match (
        covariance_conditional.as_ref(),
        certified_outer.final_hessian(),
    ) {
        (Some(v_cond), Some(outer_hessian)) => {
            let certificate = certified_outer.criterion_certificate();
            let mut excluded: Vec<usize> = certificate.lambdas_railed.clone();
            for rail in certificate.stationarity.rails() {
                if !excluded.contains(&rail.index) {
                    excluded.push(rail.index);
                }
            }
            let no_gradient = Array1::<f64>::zeros(0);
            match crate::covariance::joint_smoothing_correction(
                v_cond,
                specs,
                &label_layout,
                &rho_star,
                &inner.block_states,
                outer_hessian,
                certified_outer.final_gradient().unwrap_or(&no_gradient),
                &excluded,
            )
            .map_err(|reason| CustomFamilyError::Optimization {
                context: "fit_custom_family smoothing correction",
                reason: reason.to_string(),
            })? {
                Ok((correction, active_rank)) => (
                    Some((
                        correction,
                        gam_solve::model_types::SmoothingCorrectionMethod::FirstOrderIdentifiedSubspace {
                            active_rank,
                            rho_dimension: rho_star.len(),
                        },
                    )),
                    None,
                ),
                Err(absence) => (None, Some(absence)),
            }
        }
        (Some(_), None) if rho_star.is_empty() => (None, None),
        (Some(_), None) => {
            // The run's mint refuses a declared analytic Hessian that publishes none, so a
            // certified outer without one is exactly an undeclared Hessian, and the predicate
            // that withheld it is the absence's reason.
            let reason = outer_hessian_absence.ok_or_else(|| CustomFamilyError::Optimization {
                context: "fit_custom_family smoothing correction",
                reason: "the outer search declared an analytic rho-Hessian but the certified \
                         outer published none"
                    .to_string(),
            })?;
            log::debug!(
                "[smoothing-correction] branch=unavailable reason={reason} rho_dimension={}",
                rho_star.len(),
            );
            (
                None,
                Some(gam_solve::model_types::SmoothingCorrectionAbsence::OuterHessianUndeclared {
                    reason,
                }),
            )
        }
        (None, _) => (None, None),
    };
    install_reported_posterior_mean(
        family,
        specs,
        &mut inner,
        &mut geometry,
        reported_beta.as_ref(),
    )
    .map_err(|reason| CustomFamilyError::Optimization {
        context: "fit_custom_family reported posterior mean",
        reason: reason.to_string(),
    })?;
    let geometry = Some(geometry);
    let deviance = classical_deviance_at_mode(
        family,
        &inner.block_states,
        "fit_custom_family classical deviance",
    )?;
    // The certified point in the outer objective's own coordinates, for a later
    // fit that resumes from this model (`warm_start_from`).
    // The input fingerprint is the fit entry's to add; this layer does not see
    // the request.
    let outer_warm_start = gam_solve::model_types::OuterWarmStartRecord {
        theta: rho_star.to_vec(),
        value: Some(certified_outer.final_value()),
        beta: inner
            .block_states
            .iter()
            .flat_map(|state| state.beta.iter().copied())
            .collect(),
        input_fingerprint: None,
    };
    let mut fit = assemble_custom_family_fit_result(
        inner,
        BlockwiseFitAssembly {
            rho_physical: rho_star_physical,
            deviance,
            covariance_conditional,
            geometry,
            precomputed_edf,
            canonical: Some(&canonical),
            result_specs: raw_specs,
            penalized_objective,
            outer_iterations: certified_outer.iterations(),
            outer_gradient_norm: certified_outer.final_grad_norm(),
            criterion_certificate: Some(certified_outer.criterion_certificate().clone()),
            outer_converged: true,
            joint_log_lambdas,
            smoothing_corrected,
            smoothing_correction_absence,
            coefficient_mode_selection,
            improper_penalty_null_posterior,
        },
    )?;
    fit.artifacts.outer_warm_start = Some(outer_warm_start);
    Ok(fit)
}

enum OwnedModeProvenance<'a> {
    UserFixed,
    CertifiedOuter {
        selected_theta: &'a Array1<f64>,
        outer: &'a gam_solve::rho_optimizer::CertifiedOuterResult,
    },
}

/// Pull the family's raw-coordinate joint penalty specs back through the
/// identifiability gauge into the reduced coordinate space the inner solve and
/// outer evaluator run in (`S_red = T_fullᵀ S_raw T_full`), recomputing the
/// declared nullity on the pulled-back operator when the gauge is nontrivial.
///
/// The trait contract fixes the coordinates: joint penalties arrive in RAW
/// (pre-canonicalisation) stacked coordinates, so the pullback decision must
/// key on whether the gauge is the identity — NOT on a dimension comparison.
/// When no columns are dropped the raw and reduced totals coincide even though
/// `T_full` can still be a nontrivial rotation, and skipping `TᵀST` there
/// would smooth the wrong quadratic form (a coordinate swap with S = diag(1,2)
/// must become diag(2,1)).
///
/// Shared by the outer-optimizing entry AND the fixed-log-lambda entry
/// (#2349: the fixed path previously never consulted the joint specs, so a
/// joint-penalty family — the multinomial per-class centered carrier, whose
/// per-block penalties are EMPTY — fit completely unpenalized at fixed λ).
fn pulled_back_joint_penalty_specs<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    canonical: &gam_identifiability::canonical::CanonicalSpecs,
    reduced_total: usize,
) -> Result<Vec<gam_problem::JointPenaltySpec>, CustomFamilyError> {
    let raw_specs_joint =
        family
            .joint_penalty_specs()
            .map_err(|reason| CustomFamilyError::Optimization {
                context: "fit_custom_family joint penalty specs",
                reason,
            })?;
    let t_full = &canonical.gauge.t_full;
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
                let (evals, _) =
                    pulled
                        .eigh(Side::Lower)
                        .map_err(|e| CustomFamilyError::Optimization {
                            context: "fit_custom_family joint penalty pullback rank",
                            reason: format!(
                                "eigendecomposition of pulled-back joint penalty '{}' failed: {e}",
                                spec.label.as_deref().unwrap_or("<unlabeled>"),
                            ),
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
                // The grouping is a property of the term, not of the basis, so
                // it survives the pullback unchanged (#2579).
                group: spec.group,
            };
            out.validate()
                .map_err(|e| CustomFamilyError::ConstraintViolation {
                    reason: format!("joint penalty validation failed: {e}"),
                })?;
            Ok(out)
        })
        .collect::<Result<Vec<_>, CustomFamilyError>>()
}

fn fit_custom_family_user_fixed_log_lambdas_impl<
    F: CustomFamily + Clone + Send + Sync + 'static,
>(
    family: &F,
    raw_specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
    warm_start: Option<&CustomFamilyWarmStart>,
) -> Result<gam_solve::model_types::UnifiedFitResult, CustomFamilyError> {
    // The same channel wiring the outer entry installs (#558): without it a
    // multi-output family's shared constant columns read as cross-block aliases.
    let wired = wire_output_channels(family, raw_specs)?;
    let raw_specs: &[ParameterBlockSpec] = wired.as_deref().unwrap_or(raw_specs);
    let canonical =
        gam_identifiability::canonical::canonicalize_for_identifiability_with_operating_scalars(
            raw_specs,
            &pre_fit_coefficient_coordinates(family, raw_specs),
            pre_fit_operating_scalars(family, raw_specs)?,
        )?;
    let specs: &[ParameterBlockSpec] = &canonical.reduced_specs;
    let penalty_counts = validate_blockspecs(specs)?;
    crate::inner_blockwise_fit::refuse_non_finite_declared_joint_curvature(family, specs)?;
    let rho = flatten_log_lambdas(specs);
    let per_block = split_log_lambdas(&rho, &penalty_counts)?;
    // #2349: carry the family's joint penalty specs exactly like the outer
    // entry does — the fixed path previously never consulted them, so a
    // joint-penalty family fit completely UNPENALIZED at user-fixed λ. Each
    // joint spec's user-fixed λ is its `initial_log_lambda` (per-spec settable
    // through the family's seed API — the same vector a refusal checkpoint
    // resume carries).
    let reduced_total: usize = specs.iter().map(|s| s.design.ncols()).sum();
    let joint_specs = pulled_back_joint_penalty_specs(family, &canonical, reduced_total)?;
    let mut fixed_options = options.clone();
    let joint_log_lambdas: Option<Array1<f64>> = if joint_specs.is_empty() {
        None
    } else {
        let lambdas: Vec<f64> = joint_specs.iter().map(|s| s.initial_log_lambda).collect();
        let bundle = gam_problem::JointPenaltyBundle::new(
            std::sync::Arc::new(joint_specs),
            lambdas.clone(),
            reduced_total,
        )
        .map_err(CustomFamilyError::from)?;
        fixed_options.joint_penalties = Some(std::sync::Arc::new(bundle));
        Some(Array1::from(lambdas))
    };
    let options = &fixed_options;
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
        // The inner result holds the terminal verdict (which exit fired, the
        // KKT residual and its tolerance). Rendering only the cycle count
        // reported "did not converge after 0 cycles" for a joint-Newton solve
        // that refused inside its first cycle, and hid which guard refused it
        // (#1561). The producer used by every outer route keeps that verdict.
        return Err(inner_solve_not_converged_error(&inner, options, rho.len(), 0));
    }
    let penalized_objective = inner_penalized_objective(
        &inner,
        include_exact_newton_logdet_h(family, options),
        include_exact_newton_logdet_s(family, options),
        "custom-family fixed-log-lambda fit",
    )
    .map_err(|reason| CustomFamilyError::Optimization {
        context: "fit_custom_family_fixed_log_lambdas penalized objective",
        reason: reason.to_string(),
    })?;
    refresh_all_block_etas(family, specs, &mut inner.block_states)?;
    audit_converged_identifiability(family, raw_specs, &canonical, &inner.block_states, 0)?;
    let hessian = materialize_owned_terminal_unpenalized_hessian(
        family,
        specs,
        &inner.block_states,
        inner.joint_workspace.as_ref(),
        inner.terminal_working_sets.as_deref(),
        "custom-family fixed-log-lambda terminal Hessian",
    )
    .map_err(|reason| CustomFamilyError::Optimization {
        context: "fit_custom_family_fixed_log_lambdas terminal curvature ownership",
        reason: reason.to_string(),
    })?;
    let posterior = compute_joint_posterior(
        family,
        specs,
        &inner.block_states,
        &per_block,
        options,
        Some(&hessian),
        inner.terminal_working_sets.as_deref(),
        inner.joint_workspace.as_ref(),
        inner.terminal_likelihood_score.as_ref(),
    )
    .map_err(|reason| CustomFamilyError::Optimization {
        context: "fit_custom_family_fixed_log_lambdas terminal posterior",
        reason: format!(
            "{reason}; rho_checkpoint={:?}; no fit was assembled",
            rho.as_slice().unwrap_or(&[])
        ),
    })?;
    let JointPosteriorAssembly {
        covariance_conditional,
        mut geometry,
        reported_beta,
        improper_penalty_null_posterior,
    } = posterior;
    install_reported_posterior_mean(
        family,
        specs,
        &mut inner,
        &mut geometry,
        reported_beta.as_ref(),
    )
    .map_err(|reason| CustomFamilyError::Optimization {
        context: "fit_custom_family_fixed_log_lambdas reported posterior mean",
        reason: reason.to_string(),
    })?;
    let geometry = Some(geometry);
    let deviance = classical_deviance_at_mode(
        family,
        &inner.block_states,
        "fit_custom_family_fixed_log_lambdas classical deviance",
    )?;
    assemble_custom_family_fit_result(
        inner,
        BlockwiseFitAssembly {
            rho_physical: rho,
            deviance,
            covariance_conditional,
            geometry,
            precomputed_edf: None,
            canonical: Some(&canonical),
            result_specs: raw_specs,
            penalized_objective,
            outer_iterations: 0,
            outer_gradient_norm: None,
            criterion_certificate: None,
            outer_converged: true,
            joint_log_lambdas,
            smoothing_corrected: None,
            smoothing_correction_absence: None,
            // These entries take their mode from CustomFamilyJointHyperModeSelection,
            // which does not yet carry the rule it applied (#2661).
            coefficient_mode_selection:
                gam_solve::model_types::CoefficientModeSelection::NotRecorded,
            improper_penalty_null_posterior,
        },
    )
}

/// Fit coefficients at user-fixed smoothing strengths. No outer coordinate is
/// optimized, so a converged inner mode is the complete fit provenance.
pub fn fit_custom_family_fixed_log_lambdas<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    raw_specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
    warm_start: Option<&CustomFamilyWarmStart>,
) -> Result<gam_solve::model_types::UnifiedFitResult, CustomFamilyError> {
    fit_custom_family_user_fixed_log_lambdas_impl(family, raw_specs, options, warm_start)
}

#[derive(Clone, Copy)]
enum OwnedModeCurvatureRequirement {
    CertifiedStationaryPoint,
    CertifiedLocalMinimum,
}

/// Assemble a fixed-hyperparameter fit from the exact coefficient mode owned
/// by the terminal outer evaluation.
///
/// This consuming boundary deliberately performs no canonicalization, warm
/// restart, objective replay, or inner solve. The mode, its objective, its
/// smoothing prefix, and its returned-beta Hessian workspace are one identity;
/// changing any one of them would silently switch the Laplace branch after
/// outer optimization.
fn fit_custom_family_fixed_log_lambdas_from_owned_mode_with_provenance<
    F: CustomFamily + Clone + Send + Sync + 'static,
>(
    family: &F,
    specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
    mode: CustomFamilyOwnedMode,
    provenance: OwnedModeProvenance<'_>,
    curvature_requirement: OwnedModeCurvatureRequirement,
) -> Result<gam_solve::model_types::UnifiedFitResult, CustomFamilyError> {
    let (outer_iterations, outer_gradient_norm, criterion_certificate, certified_theta, certified_outer) =
        match provenance {
            OwnedModeProvenance::UserFixed => (0, None, None, None, None),
            OwnedModeProvenance::CertifiedOuter {
                selected_theta,
                outer,
            } => {
                // The curvature question is answered by the certificate's own
                // verdict, the object the outer search accepted the point on —
                // not by the raw analytic flag. An analytic negative direction the
                // criterion CONTRADICTED (probed along its eigenvector at every
                // scale down to the criterion's resolution and found not to
                // descend, #2612) is a minimum as far as the objective can tell;
                // refusing it on the flag alone took a fit the outer had
                // certified and threw it away one call later (gam#2765).
                // Inadmissible curvature and unevaluated curvature are still
                // refused: the first is a saddle, the second is no evidence.
                if matches!(
                    curvature_requirement,
                    OwnedModeCurvatureRequirement::CertifiedLocalMinimum
                ) && !matches!(
                    outer.criterion_certificate().curvature_verdict(),
                    gam_solve::model_types::CurvatureAdmissibility::Admissible
                        | gam_solve::model_types::CurvatureAdmissibility::CriterionContradicted
                ) {
                    return Err(CustomFamilyError::Optimization {
                        context:
                            "fit_custom_family_fixed_log_lambdas_from_owned_mode outer curvature",
                        reason: format!(
                            "a profiled nonconvex coefficient mode requires an outer curvature certificate that admits a local minimum; this one is {}",
                            outer.criterion_certificate().curvature_verdict()
                        ),
                    });
                }
                if selected_theta.len() != outer.rho().len()
                    || selected_theta
                        .iter()
                        .zip(outer.rho().iter())
                        .any(|(selected, certified)| selected.to_bits() != certified.to_bits())
                {
                    return Err(CustomFamilyError::InvalidInput {
                        context:
                            "fit_custom_family_fixed_log_lambdas_from_owned_mode outer identity",
                        reason: "the selected full hyperparameter vector does not bitwise match the certified outer optimum"
                            .to_string(),
                    });
                }
                (
                    outer.iterations(),
                    outer.final_grad_norm(),
                    Some(outer.criterion_certificate().clone()),
                    Some((outer.rho(), outer.final_value())),
                    Some(outer),
                )
            }
        };

    let CustomFamilyOwnedMode {
        objective: selected_objective,
        rho,
        hyper_values,
        mut inner,
        psi_scores,
    } = mode;
    if !inner.converged {
        return Err(CustomFamilyError::Optimization {
            context: "fit_custom_family_fixed_log_lambdas_from_owned_mode",
            reason: "the selected coefficient branch was not converged; no fit was assembled"
                .to_string(),
        });
    }

    if let Some((_, certified_objective)) = certified_theta
        && selected_objective.to_bits() != certified_objective.to_bits()
    {
        return Err(CustomFamilyError::Optimization {
            context: "fit_custom_family_fixed_log_lambdas_from_owned_mode objective identity",
            reason: format!(
                "selected profile objective does not belong to the certified outer optimum: selected={selected_objective:.17e}, certified={certified_objective:.17e}",
            ),
        });
    }

    if let Some((certified_theta, _)) = certified_theta {
        let expected_len = rho.len() + hyper_values.len();
        let identity_matches = certified_theta.len() == expected_len
            && certified_theta
                .iter()
                .zip(rho.iter().chain(hyper_values.iter()))
                .all(|(certified, selected)| certified.to_bits() == selected.to_bits());
        if !identity_matches {
            return Err(CustomFamilyError::InvalidInput {
                context: "fit_custom_family_fixed_log_lambdas_from_owned_mode full hyper identity",
                reason: "the owned mode's [rho | manifest values] do not bitwise match the certified outer optimum"
                    .to_string(),
            });
        }
    }
    let spec_rho = flatten_log_lambdas(specs);
    if rho.len() != spec_rho.len()
        || rho
            .iter()
            .zip(spec_rho.iter())
            .any(|(selected, configured)| selected.to_bits() != configured.to_bits())
    {
        return Err(CustomFamilyError::InvalidInput {
            context: "fit_custom_family_fixed_log_lambdas_from_owned_mode",
            reason: "selected smoothing coordinates do not bitwise match the supplied coefficient-mode geometry"
                .to_string(),
        });
    }
    let penalty_counts = validate_blockspecs(specs)?;
    crate::inner_blockwise_fit::refuse_non_finite_declared_joint_curvature(family, specs)?;
    let per_block = split_log_lambdas(&rho, &penalty_counts)?;
    // Audit the geometry the outer entry audits: a family's declared output
    // channels are installed first (#558). Without them the latent survival
    // family's constant log-σ column reads as an alias of the mean intercept and
    // a certified mode is refused as still needing reduction (#2714).
    let wired = wire_output_channels(family, specs)?;
    let audit_specs: &[ParameterBlockSpec] = wired.as_deref().unwrap_or(specs);
    let canonical =
        gam_identifiability::canonical::canonicalize_for_identifiability_with_operating_scalars(
            audit_specs,
            &pre_fit_coefficient_coordinates(family, audit_specs),
            pre_fit_operating_scalars(family, audit_specs)?,
        )?;
    if !canonical.gauge.is_identity()
        || canonical.reduced_specs.len() != specs.len()
        || canonical
            .reduced_specs
            .iter()
            .zip(specs.iter())
            .any(|(reduced, exact)| reduced.design.ncols() != exact.design.ncols())
    {
        return Err(CustomFamilyError::InvalidInput {
            context: "fit_custom_family_fixed_log_lambdas_from_owned_mode canonical geometry",
            reason: "the terminal outer evaluator returned a mode in a coefficient geometry that still requires identifiability reduction; exact outer evaluation must own the canonical geometry before certification"
                .to_string(),
        });
    }
    audit_converged_identifiability(
        family,
        audit_specs,
        &canonical,
        &inner.block_states,
        outer_iterations,
    )?;

    let hessian = materialize_owned_terminal_unpenalized_hessian(
        family,
        specs,
        &inner.block_states,
        inner.joint_workspace.as_ref(),
        inner.terminal_working_sets.as_deref(),
        "selected-mode final Hessian",
    )
    .map_err(|reason| CustomFamilyError::Optimization {
        context: "fit_custom_family_fixed_log_lambdas_from_owned_mode curvature identity",
        reason: reason.to_string(),
    })?;

    let posterior = compute_joint_posterior(
        family,
        specs,
        &inner.block_states,
        &per_block,
        options,
        Some(&hessian),
        inner.terminal_working_sets.as_deref(),
        inner.joint_workspace.as_ref(),
        inner.terminal_likelihood_score.as_ref(),
    )
    .map_err(|reason| CustomFamilyError::Optimization {
        context: "fit_custom_family_fixed_log_lambdas_from_owned_mode posterior",
        reason: reason.to_string(),
    })?;
    let JointPosteriorAssembly {
        covariance_conditional,
        mut geometry,
        reported_beta,
        improper_penalty_null_posterior,
    } = posterior;
    // #2677: first-order hyperparameter-uncertainty correction over the full
    // certified outer coordinate θ = [ρ | ψ], from the SAME analytic outer
    // Hessian the certificate judged and the ψ scores the owning evaluation
    // assembled. A user-fixed mode has no outer uncertainty to propagate.
    let (smoothing_corrected, smoothing_correction_absence) = match (
        covariance_conditional.as_ref(),
        certified_outer,
    ) {
        (Some(v_cond), Some(outer)) if !rho.is_empty() || !hyper_values.is_empty() => {
            let theta_dimension = rho.len() + hyper_values.len();
            let psi_dimension = hyper_values.len();
            match (outer.final_hessian(), psi_scores.as_ref()) {
                (None, _) => (
                    None,
                    Some(
                        gam_solve::model_types::SmoothingCorrectionAbsence::OuterHessianUndeclared {
                            reason: gam_solve::model_types::OuterHessianAbsence::NotPublished,
                        },
                    ),
                ),
                (Some(_), None) if psi_dimension > 0 => (
                    None,
                    Some(
                        gam_solve::model_types::SmoothingCorrectionAbsence::FamilyHyperScoresUnrecorded {
                            psi_dimension,
                        },
                    ),
                ),
                (Some(outer_hessian), recorded) => {
                    let no_psi_scores = Array2::<f64>::zeros((v_cond.nrows(), 0));
                    let scores = recorded.unwrap_or(&no_psi_scores);
                    if scores.ncols() != psi_dimension {
                        return Err(CustomFamilyError::DimensionMismatch {
                            reason: format!(
                                "owned-mode smoothing correction: {} psi score column(s) for {psi_dimension} family hyperparameter(s)",
                                scores.ncols()
                            ),
                        });
                    }
                    let certificate = outer.criterion_certificate();
                    let mut excluded: Vec<usize> = certificate.lambdas_railed.clone();
                    for rail in certificate.stationarity.rails() {
                        if !excluded.contains(&rail.index) {
                            excluded.push(rail.index);
                        }
                    }
                    let no_gradient = Array1::<f64>::zeros(0);
                    match crate::covariance::owned_mode_smoothing_correction(
                        v_cond,
                        specs,
                        &rho,
                        &inner.block_states,
                        scores,
                        outer_hessian,
                        outer.final_gradient().unwrap_or(&no_gradient),
                        &excluded,
                    )? {
                        Ok((correction, active_rank)) => (
                            Some((
                                correction,
                                gam_solve::model_types::SmoothingCorrectionMethod::FirstOrderIdentifiedSubspace {
                                    active_rank,
                                    rho_dimension: theta_dimension,
                                },
                            )),
                            None,
                        ),
                        Err(absence) => (None, Some(absence)),
                    }
                }
            }
        }
        _ => (None, None),
    };
    install_reported_posterior_mean(
        family,
        specs,
        &mut inner,
        &mut geometry,
        reported_beta.as_ref(),
    )
    .map_err(|reason| CustomFamilyError::Optimization {
        context: "fit_custom_family_fixed_log_lambdas_from_owned_mode reported posterior mean",
        reason: reason.to_string(),
    })?;
    let geometry = Some(geometry);

    let deviance = classical_deviance_at_mode(
        family,
        &inner.block_states,
        "fit_custom_family_fixed_log_lambdas classical deviance",
    )?;
    assemble_custom_family_fit_result(
        inner,
        BlockwiseFitAssembly {
            rho_physical: rho,
            deviance,
            covariance_conditional,
            geometry,
            precomputed_edf: None,
            canonical: Some(&canonical),
            result_specs: specs,
            penalized_objective: selected_objective,
            outer_iterations,
            outer_gradient_norm,
            criterion_certificate,
            outer_converged: true,
            joint_log_lambdas: None,
            smoothing_corrected,
            smoothing_correction_absence,
            // These entries take their mode from CustomFamilyJointHyperModeSelection,
            // which does not yet carry the rule it applied (#2661).
            coefficient_mode_selection:
                gam_solve::model_types::CoefficientModeSelection::NotRecorded,
            improper_penalty_null_posterior,
        },
    )
}

fn owned_mode_from_selection(
    selection: CustomFamilyJointHyperModeSelection,
) -> Result<CustomFamilyOwnedMode, CustomFamilyError> {
    let CustomFamilyJointHyperModeSelection {
        result,
        selected_candidate,
        screened_objectives,
        mode,
        ..
    } = selection;
    if !result.inner_converged || !mode.inner.converged {
        return Err(CustomFamilyError::Optimization {
            context: "owned coefficient-mode selection",
            reason: "the selected coefficient branch was not converged".to_string(),
        });
    }
    let screened_objective = screened_objectives
        .get(selected_candidate)
        .and_then(|objective| *objective)
        .ok_or_else(|| CustomFamilyError::Optimization {
            context: "owned coefficient-mode selection objective identity",
            reason: "the selected candidate has no finite screened profile objective".to_string(),
        })?;
    if screened_objective.to_bits() != result.objective.to_bits()
        || result.objective.to_bits() != mode.objective.to_bits()
    {
        return Err(CustomFamilyError::Optimization {
            context: "owned coefficient-mode selection objective identity",
            reason: format!(
                "selected profile objective changed across screening/result/mode ownership: screened={screened_objective:.17e}, result={:.17e}, mode={:.17e}",
                result.objective, mode.objective,
            ),
        });
    }
    let carried = &result.warm_start.inner;
    let rho_matches = carried.rho.len() == mode.rho.len()
        && carried
            .rho
            .iter()
            .zip(mode.rho.iter())
            .all(|(left, right)| left.to_bits() == right.to_bits());
    let beta_matches = carried.block_beta.len() == mode.inner.block_states.len()
        && carried
            .block_beta
            .iter()
            .zip(mode.inner.block_states.iter())
            .all(|(left, state)| {
                left.len() == state.beta.len()
                    && left
                        .iter()
                        .zip(state.beta.iter())
                        .all(|(left, right)| left.to_bits() == right.to_bits())
            });
    if !rho_matches || !beta_matches {
        return Err(CustomFamilyError::Optimization {
            context: "owned coefficient-mode selection state identity",
            reason: "the selected public result and owned inner mode have different rho or coefficient bits"
                .to_string(),
        });
    }
    Ok(mode)
}

/// Assemble a caller-fixed coefficient mode. No outer coordinate was
/// optimized, so the resulting fit deliberately carries no outer certificate.
pub fn fit_custom_family_user_fixed_log_lambdas_from_mode_selection<
    F: CustomFamily + Clone + Send + Sync + 'static,
>(
    family: &F,
    specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
    selection: CustomFamilyJointHyperModeSelection,
) -> Result<gam_solve::model_types::UnifiedFitResult, CustomFamilyError> {
    let mode = owned_mode_from_selection(selection)?;
    fit_custom_family_fixed_log_lambdas_from_owned_mode_with_provenance(
        family,
        specs,
        options,
        mode,
        OwnedModeProvenance::UserFixed,
        OwnedModeCurvatureRequirement::CertifiedLocalMinimum,
    )
}

/// Assemble the coefficient mode that belongs bit-for-bit to a certified
/// second-order outer optimum.
pub fn fit_custom_family_fixed_log_lambdas_from_mode_selection<
    F: CustomFamily + Clone + Send + Sync + 'static,
>(
    family: &F,
    specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
    selection: CustomFamilyJointHyperModeSelection,
    selected_theta: &Array1<f64>,
    outer: &gam_solve::rho_optimizer::CertifiedOuterResult,
) -> Result<gam_solve::model_types::UnifiedFitResult, CustomFamilyError> {
    let mode = owned_mode_from_selection(selection)?;
    fit_custom_family_fixed_log_lambdas_from_owned_mode_with_provenance(
        family,
        specs,
        options,
        mode,
        OwnedModeProvenance::CertifiedOuter {
            selected_theta,
            outer,
        },
        OwnedModeCurvatureRequirement::CertifiedLocalMinimum,
    )
}

/// Assemble the exact coefficient mode installed by the outer optimizer's
/// terminal full-data evaluation. No optimizer, objective replay, or inner
/// coefficient solve is entered at this boundary.
pub fn fit_custom_family_fixed_log_lambdas_from_owned_mode<
    F: CustomFamily + Clone + Send + Sync + 'static,
>(
    family: &F,
    specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
    mode: CustomFamilyOwnedMode,
    selected_theta: &Array1<f64>,
    outer: &gam_solve::rho_optimizer::CertifiedOuterResult,
) -> Result<gam_solve::model_types::UnifiedFitResult, CustomFamilyError> {
    fit_custom_family_fixed_log_lambdas_from_owned_mode_with_provenance(
        family,
        specs,
        options,
        mode,
        OwnedModeProvenance::CertifiedOuter {
            selected_theta,
            outer,
        },
        OwnedModeCurvatureRequirement::CertifiedStationaryPoint,
    )
}

pub fn fit_custom_family_fixed_log_lambda_warm_start<
    F: CustomFamily + Clone + Send + Sync + 'static,
>(
    family: &F,
    raw_specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
) -> Result<(Vec<Array1<f64>>, bool, usize), CustomFamilyError> {
    // The same channel wiring the outer entry installs (#558).
    let wired = wire_output_channels(family, raw_specs)?;
    let raw_specs: &[ParameterBlockSpec] = wired.as_deref().unwrap_or(raw_specs);
    let canonical =
        gam_identifiability::canonical::canonicalize_for_identifiability_with_operating_scalars(
            raw_specs,
            &pre_fit_coefficient_coordinates(family, raw_specs),
            pre_fit_operating_scalars(family, raw_specs)?,
        )?;
    let specs: &[ParameterBlockSpec] = &canonical.reduced_specs;
    let penalty_counts = validate_blockspecs(specs)?;
    crate::inner_blockwise_fit::refuse_non_finite_declared_joint_curvature(family, specs)?;
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

