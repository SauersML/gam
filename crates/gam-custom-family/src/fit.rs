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
        log::info!(
            "[CANON] structural coefficient coordinate(s) declared: [{}] — these blocks keep \
             their basis AND their width through canonicalisation (#2748)",
            structural.join(", "),
        );
    }
    coordinates
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
    let (beta_pilot, beta_current) = drift_audit_beta_pair(raw_specs, &raw_states);
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
            log::info!(
                "[AUDIT-TRANSPORT] converged identifiability accepted on ENDPOINT AGREEMENT \
                 ONLY (rank={}): the pilot certificate's transport radius {radius:.3e} was \
                 exhausted by an excursion of at least {excursion:.3e}, so the pilot verdict is \
                 not certified along the path it travelled",
                drift.current_rank
            );
        }
        Some(true) => {
            let (excursion, radius) = drift.excursion_vs_radius.unwrap_or((f64::NAN, f64::NAN));
            log::debug!(
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
    pub(crate) precomputed_edf: Option<(f64, Vec<f64>, Vec<f64>, Vec<f64>)>,
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

    blockwise_fit_from_parts(
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

/// Fraction of a rank-deficient term's ATTAINABLE df that the floor retains
/// (#2608).
///
/// [`EFFECTIVE_DF_FLOOR`] is absolute, and a rank-1 penalty's structural edf
/// `γ/(γ + e^ρ)` ranges over `(0, 1]` — it reaches 1 only as `λ → 0`. So the
/// absolute floor is UNREACHABLE for every rank-1 term and the guard skipped
/// them all. The null-space half of a Marra–Wood double penalty is rank-1, so
/// the LINEAR direction of every smooth was exempt from the only protection
/// against its own collapse: measured on penguins at `edf = 4.6e-5`, dead to
/// five decimals, while `nnet::multinom` scores 0.9912 there with a purely
/// LINEAR softmax.
///
/// A term that cannot reach the absolute floor is instead asked to retain this
/// fraction of what it CAN reach. For rank 1 the crossing is closed form,
/// `ρ*(f) = ln γ + ln((1 − f)/f)`, finite for every `f ∈ (0, 1)` — so the
/// relative floor is well posed exactly where the absolute one is not, and it is
/// γ-ADAPTIVE rather than a uniform wall: a well-supported direction (`γ = 1e4`)
/// is held only 2.8 nats below the ceiling while a weakly supported one
/// (`γ = 1e-2`) is held 16.6. The fraction sets how much of the attainable df to
/// keep; the DATA sets where that lands.
///
/// This can only TIGHTEN, and only for terms that were previously exempt: when
/// `edf_max > EFFECTIVE_DF_FLOOR` the target is still the absolute floor, so
/// every term the guard already bounded is byte-identical.
///
/// Chosen on evidence, not picked, and the evidence is a measured curve rather
/// than an argument. At `0.5` the penguins LINEAR direction came back and
/// accuracy went `0.8421 → 0.9649` (#2579), leaving one failing assertion:
/// held-out log-loss against `nnet`'s `0.09494`, i.e. the right class picked
/// under-confidently (#2612). #2612 pre-registered a sweep and what it would
/// mean. Three arms off one base commit, differing ONLY in this constant, on
/// `gam_multinomial_classifies_penguin_species_at_least_as_well_as_nnet_on_real_data`:
///
/// ```text
///   f     log-loss   accuracy   per-class recall
///   0.50   0.26080    0.9649    [0.961, 0.909, 1.000]
///   0.75   0.17614    0.9649    [0.961, 0.909, 1.000]
///   0.85   0.14087    0.9649    [0.961, 0.909, 1.000]     bar = 0.14494
///   0.90   0.12057    0.9649    [0.961, 0.909, 1.000]   <- chosen
/// ```
///
/// Monotone, 0.140 nats over the sweep. Accuracy and every per-class recall are
/// INVARIANT at every `f`, so this buys calibration without trading
/// classification — the failure really was residual shrinkage of the linear
/// direction, which is reading (1) of the pre-registration.
///
/// THE CURVE ALONE DOES NOT CHOOSE THIS CONSTANT, because the sweep above
/// watches ONE train/test split and `f` has a second effect it cannot see.
/// Capping ρ lower means less shrinkage, hence a WIDER Laplace posterior, and at
/// `f = 0.90` a sibling split (stride 4) stopped PREDICTING outright —
/// `logistic-normal quadrature did not converge through Smolyak level 12` —
/// against a same-base `f = 0.75` control where it passes. That is attributable
/// to this constant, and it is a hard failure rather than a metric regression.
///
/// So this value was bounded on BOTH sides, by different things: the log-loss bar
/// is crossed near `f ≈ 0.83`, and the quadrature stopped certifying in
/// `(0.85, 0.90]`. `0.85` sat in that window with only 0.004 nats of margin.
///
/// The upper bound is now GONE rather than respected. The failing fit had spent
/// 9633 of its 2000000 evaluations, so the LEVEL ceiling was binding and not the
/// cost guard; raising it to 16 (see `MultinomialPosteriorIntegrationControl`,
/// the same move #2350 made from 8) lets that split certify. Measured at
/// `f = 0.90` with the raised ceiling, in one run, BOTH arms pass:
///
/// ```text
///   stride-3   log-loss 0.12057   vs nnet 0.09494   (bar 0.14494)
///   stride-4   log-loss 0.17246   vs nnet 0.76930
/// ```
///
/// Strictly better than `0.85` on both — 0.12057 against 0.14087, and 0.17246
/// against 0.20708 — and the stride-3 margin goes 0.004 → 0.024 nats, which is
/// the difference between a bar that holds and one that flips on noise. It costs
/// wall clock: that pair ran 515s against 244s, because a wider posterior really
/// does visit the deeper levels. A well-conditioned fit certifies early and never
/// pays it.
///
/// `ρ*(0.90) = ln γ − 2.20` is finite and moderate; it is `f → 1` that sends `ρ*`
/// to `−∞`, and this stays a wide margin short of it.
///
/// What this does NOT fix, measured rather than assumed: `predict_multinomial_formula`
/// publishes `E[softmax(η)]` while `nnet` publishes `softmax(η̂)`, and the same
/// fit reports `0.17614` posterior-mean against `0.16499` plug-in at `f = 0.75`.
/// Posterior width therefore accounts for `0.011` nats — about an eighth of the
/// gap that remained there. The rest was the mode, which is why moving this
/// constant is the right instrument and not a coincidence.
/// WHAT THIS CONSTANT IS, in closed form (#2615).
///
/// For a rank-1 penalty the structural edf is `edf(ρ) = γ/(γ + e^ρ)` and
/// `edf_max = unit_weight_term_edf_at_physical_strength(gammas, 0.0) = 1`
/// EXACTLY — at `λ = 0` every positive `γ_j` contributes `1/(1 + 0) = 1`, so
/// `edf_max` is the penalty rank and carries no design dependence at all. The
/// relative target is therefore just `f`, and the bisected bound has an exact
/// solution:
///
/// ```text
///   γ/(γ + e^{ρ*}) = f    ⟺    ρ* = ln γ + ln((1 − f)/f)
/// ```
///
/// which is why `f = 0.90` records `ρ* = ln γ − 2.20`: `ln(1/9) = −2.1972`.
/// The bisection is not discovering a shape; it is recovering a logit.
///
/// AND WHAT `f` MEANS. `edf = γ/(γ + λ)` is the DATA's share of the posterior
/// precision along that direction — `γ` is the design information, `λ` the
/// prior precision. So the floor
///
/// ```text
///   edf ≥ f    ⟺    λ/γ ≤ (1 − f)/f
/// ```
///
/// is a ceiling on the PRIOR-TO-DATA PRECISION ODDS for the linear direction of
/// a smooth. That is the derivation this constant was missing: it is not a
/// fraction of an arbitrary quantity, it is an odds ratio, and it should be
/// argued as one.
///
///   f = 0.50  ⟺  odds 1:1 — the data merely outvotes the prior. This is the
///                identifiability threshold, and it is where #2608 started.
///   f = 0.90  ⟺  odds 1:9 — the prior may carry at most a tenth of the
///                precision. A much stronger demand than identifiability.
///
/// The odds reading also explains the `f → 1 ⟹ ρ* → −∞` behaviour noted below
/// without appeal to the fit: odds → 0 means no prior mass at all is admissible,
/// which is `λ = 0`.
///
/// HONEST STATUS. The FORM above is derived. The VALUE is not: `0.90` was
/// selected as the largest value a raised Smolyak ceiling would admit while
/// beating a log-loss bar on penguins, which is a different claim from "1:9 is
/// the right odds for a linear trend". Recorded here rather than left implied,
/// because those two statements have very different standing and the code used
/// to read as the second. Deriving the value is #2615; the measurements below
/// are what is actually known.
/// RETRACTED 2026-07-31 (#2612), BY RE-MEASUREMENT AT THE EXACT QUADRATURE.
/// Two claims above are refuted, and by one cause: every number in the
/// `0.5 → 0.85 → 0.90` record was produced through the isotropic Smolyak path
/// that #2612 subsequently showed was NOT converging on this posterior (off by
/// four orders at level 16; it went on to refuse outright at the same settings).
/// The exact conditioned three-class quadrature that replaced it disagrees.
/// Re-measured on the same fixture, same split, at `c98f0b0d6`:
///
///  * "Posterior width therefore accounts for `0.011` nats." It accounts for
///    `0.137102`. The same fit reports `0.161820` posterior-mean against
///    `0.024718` plug-in — a factor of 12.5. Do not cite `0.011` again.
///
///  * "Moving this constant is the right instrument and not a coincidence." A
///    same-base control arm at `f = 0.50` — the value #2612 was opened at, and
///    the far end of the recorded `0.26080 → 0.12057` curve — measures held-out
///    log-loss `0.160210`, against `0.161820` at `f = 0.90`. That is `0.0016`
///    nats, not `0.14`. **On this fixture this constant is INERT**, and the
///    curve it was selected from does not exist at the exact quadrature.
///
/// WHY AN UPPER BOUND CANNOT BITE HERE, which is the part worth keeping. This
/// floor manufactures an UPPER ρ bound. On penguins four of the eight live
/// null-space λ sit EXACTLY on the ρ box's LOWER wall, `2.173913043e-4` — which
/// is `multinomial_formula_min_lambda` on this split to every digit
/// (`8.0e-4 × 0.25 × 50/46`, with 46 the training Chinstrap count). REML is
/// railed at the least-smoothed value the box permits and is pushing further, so
/// a ceiling sitting above it never binds. The issue's framing — and this
/// constant — are about OVER-smoothing; this fixture is railed at the UNDER-
/// smoothed wall.
///
/// The constant this fixture actually reads out is
/// `MULTINOMIAL_FORMULA_PRIOR_PSEUDO_OBS` in `gam-models::multinomial`, where
/// the measurement is recorded. None of this makes `0.90` wrong; it makes the
/// evidence for it void, so the value stands unsupported rather than supported,
/// and it is NOT re-derivable from the sweep above.
pub(crate) const EFFECTIVE_DF_FLOOR_RELATIVE_FRACTION: f64 = 0.90;

/// Uniform ρ = log λ over-smoothing ceiling for the custom-family outer box, on
/// top of which each term's per-coordinate `EFFECTIVE_DF_FLOOR` bound is
/// tightened. Two forces bracket it:
///
///  * FROM BELOW — legitimate REML optima. A smooth mean over a genuinely smooth
///    signal wants heavy shrinkage: the #1561 Gaussian location-scale `s(x,
///    bs='tp')` mean over `sin(2πx)` has its REML optimum at ρ ≈ 11 (edf ≈ 15).
///    The former `10.0` ceiling clipped exactly that — the μ coordinate railed at
///    ρ = log λ = 10.0 = e¹⁰, the outer bound-projection zeroed its (still −3.5)
///    gradient, and the fit certified a spurious constrained optimum at edf ≈ 19,
///    leaving the mean under-smoothed (#1561/#2356). The ceiling MUST sit above
///    the heavy-but-finite optima the data legitimately selects, matching the
///    over-smoothing range the seed prepass itself already explores
///    (`crate::estimate::RHO_BOUND`, optimizer.rs).
///  * FROM ABOVE — numerical stability. Beyond λ ≈ 10⁹ (ρ ≈ 20.7) the profiled
///    criterion goes dead-flat, ARC's quadratic model degrades, and the
///    retry-stall / empty-`block_states` failure paths surface. The ceiling stays
///    a wide margin below that region.
///
/// `12.0` (λ ≈ 163k) is the smallest raise that clears the #1561 mean-smooth
/// optimum (ρ_μ ≈ 11.06 on the plain arm) with real headroom, and it matches the
/// value the spatial exact-joint location-scale path already boxes ρ to
/// (`fit_orchestration::drivers::JOINT_RHO_BOUND`, its joint-search prior) — a regime that path fits
/// stably. Pushing the uniform ceiling further (e.g. 15) let some delicate
/// wiggle / real-data tp location-scale fits (gagurine, the spatial
/// engine↔reference parity fixtures) explore a warm-start/inner-solve path where
/// the joint PIRLS stopped converging, so 12 keeps the raise tight. The per-term
/// `EFFECTIVE_DF_FLOOR` bound — not this uniform cap — is what protects a term
/// from collapsing onto its unpenalized null space, so this only frees the
/// coordinates whose honest optimum was being clipped at ρ = 10.
///
/// #1561 forensic note (2026-07-26): for a location-scale fit with NO spatial
/// terms, the exact-joint (ρ, ψ) outer optimizer is inactive
/// (`log_kappa_dim() == 0`), the fit routes through `fit_custom_family`, and
/// THIS ceiling — not `JOINT_RHO_BOUND` — is the operative ρ box. A
/// selected λ equal to `exp(EFFECTIVE_DF_CEILING)` to the last bit is a
/// coordinate railed HERE (`ln λ = 12.000000…` ⇒ check this constant first);
/// widening the location-scale engine's box cannot move it, and was measured
/// not to (bit-identical λ across a 5× widening of that box). The value
/// coincidence with `JOINT_RHO_BOUND` is what makes the misattribution
/// cheap. A null-space-ridge coordinate railed here with z² = θ̂²g ≤ 1 is
/// HARMLESS — the criterion is monotone to +∞ and the shrinkage factor
/// g/(g+e¹²) ≈ 1e-4 means the λ=∞ limit is already attained — while a
/// Primary railed here with a finite beyond-ceiling optimum is the #2356
/// class. A ceiling sweep 12→14→20 on the #1561 by-group fixture moved every
/// truth-RMSE by < 1e-5, both cases included.
///
/// Exported `pub` because regimes that PIN a coordinate at the strong-smoothing
/// wall seed from it (the survival parametric-AFT time-warp seed, #2356): a
/// wall-pinning seed must move WITH the ceiling, or a ceiling raise strands it
/// interior and re-opens the flat-ridge crawl the seed exists to kill. Seeding
/// AT this ceiling is exact even when a term's realized upper bound is tighter
/// (the `EFFECTIVE_DF_FLOOR` tightening): `run_plan` projects every seed onto
/// the realized per-coordinate box, so "seed = ceiling" lands ON the wall.
pub const EFFECTIVE_DF_CEILING: f64 = 12.0;

/// The lower wall of the outer ρ box — the caller's
/// [`BlockwiseFitOptions::rho_lower_bound`] (default `-10.0`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct RhoLowerWall(pub(crate) f64);

/// The uniform over-smoothing ceiling of the outer ρ box —
/// [`EFFECTIVE_DF_CEILING`] (`12.0`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct RhoCeiling(pub(crate) f64);

/// The admissible outer ρ = log λ interval, carried as ONE validated value so
/// its two walls can neither be transposed nor drift apart.
///
/// The walls are independently owned: the floor is the caller's
/// `rho_lower_bound` (default `-10.0`) and the ceiling is
/// [`EFFECTIVE_DF_CEILING`] (`12.0`). #2370 was precisely these two constants
/// drifting apart — #2356 raised the ceiling `10 → 12` while the floor stayed
/// at `-10`, opening a window in which the derived per-term upper bound could
/// land *below* the floor and invert the box, whose `f64::clamp(min, max)`
/// then panicked across the FFI boundary.
///
/// The follow-up hazard is the one this type closes. Passing the same two
/// walls as adjacent bare `f64` parameters let a caller hand them over
/// BACKWARDS with no compile error: the transposed call produced a
/// plausible-looking typed error at runtime instead, and a real regression
/// test was observed doing exactly that against the landed signature. Wrapping
/// each wall in its own newtype makes the transposition a type error, and
/// funnelling both through one checked constructor means the ordering
/// invariant is established once rather than restated at every call site.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct RhoBox {
    lower: f64,
    ceiling: f64,
}

impl RhoBox {
    /// Build the box, rejecting a non-finite wall or an inverted interval.
    ///
    /// The empty case is `lower > ceiling` and nothing else. A box with
    /// `lower == ceiling` is a legal PINNED coordinate: the caller has fixed λ
    /// rather than asked for an impossible range, and the derivation handles it
    /// without a special case — no tightening is possible, so the term keeps the
    /// uniform ceiling and the emitted box stays well-ordered.
    ///
    /// This deliberately matches the outer optimizer, which accepts a pinned
    /// coordinate (`rho_optimizer::run_plan_tests::pinned_equal_rho_bounds_are_accepted_2370`).
    /// Refusing here what the layer below accepts would be a cross-layer
    /// contract split — the same class of defect as #2370 itself, one level up:
    /// two independently-owned definitions of the same admissible set, free to
    /// drift apart.
    pub(crate) fn new(
        RhoLowerWall(lower): RhoLowerWall,
        RhoCeiling(ceiling): RhoCeiling,
    ) -> Result<Self, CustomFamilyError> {
        gam_problem::validate_log_strength(ceiling).map_err(|error| {
            CustomFamilyError::InvalidInput {
                context: "effective-DF rho ceiling",
                reason: error.to_string(),
            }
        })?;
        gam_problem::validate_log_strength(lower).map_err(|error| {
            CustomFamilyError::InvalidInput {
                context: "effective-DF rho lower bound",
                reason: error.to_string(),
            }
        })?;
        if lower > ceiling {
            return Err(CustomFamilyError::InvalidInput {
                context: "effective-DF rho box",
                reason: format!(
                    "rho lower bound {lower} exceeds the uniform ceiling {ceiling}; \
                     the admissible ρ-box is empty"
                ),
            });
        }
        Ok(Self { lower, ceiling })
    }

    pub(crate) fn lower(&self) -> f64 {
        self.lower
    }

    pub(crate) fn ceiling(&self) -> f64 {
        self.ceiling
    }
}

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
    penalty_range_gammas_from_gram(&gram, &s_dense)
}

/// The pencil core of [`design_penalty_range_gammas`], reading the unit-weight
/// Gram directly instead of a [`DesignMatrix`].
///
/// A [`gam_problem::JointPenaltySpec`] has no `DesignMatrix` of its own: its
/// matrix spans the CONCATENATION of every block's coefficients, so there is no
/// single block whose design it multiplies. Splitting the core out here is what
/// lets a joint penalty reach the identical structural-edf machinery, with
/// [`stacked_block_design_gram`] supplying the matching Gram (#2579).
pub(crate) fn penalty_range_gammas_from_gram(
    gram: &Array2<f64>,
    s_dense: &Array2<f64>,
) -> Option<Vec<f64>> {
    let p = s_dense.nrows();
    if p == 0 || s_dense.ncols() != p || gram.nrows() != p || gram.ncols() != p {
        return None;
    }
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
    let mut b = y.t().dot(gram).dot(&y);
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
        let g00 = u0.t().dot(gram).dot(&u0); // r0×r0
        let g_r0 = y.t().dot(gram).dot(&u0); // r×r0, already D_r^{-1/2}-scaled rows
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
    rho_box: RhoBox,
) -> Result<Array1<f64>, CustomFamilyError> {
    // The edf-floor tightening must be evaluated against the SAME lower wall the
    // optimizer will actually enforce (`options.rho_lower_bound`), not against a
    // `-ceiling` proxy. The two are independent constants and are NOT equal in
    // production: the uniform ceiling is `EFFECTIVE_DF_CEILING = 12` while the
    // default `rho_lower_bound = -10`. Bisecting for / guarding the edf=1
    // crossing against `-ceiling = -12` while the box floor sits at `-10` lets
    // the crossing land in (-12, -10) and emits an upper bound BELOW the lower
    // bound — an inverted ρ-box whose `f64::clamp(min, max)` in
    // `project_to_bounds` panics with `min > max` across the FFI boundary
    // (#2370). Anchoring every check on `lower` keeps the emitted upper bound
    // strictly above the floor by construction.
    //
    // Both walls arrive inside [`RhoBox`], which has already established that
    // they are finite and correctly ordered, so this body can read them
    // without re-validating and no caller can transpose them.
    let ceiling = rho_box.ceiling();
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
            tighten_effective_df_floor_bound(&gammas, rho_box, &mut upper[outer])?;
        }
    }
    // #2579: for the one family that carries JOINT penalties, the loop above
    // iterates nothing. `#1587` emptied the multinomial's per-block `penalties`
    // lists — the centered `M ⊗ S_t` bundle became the sole smoothing carrier —
    // so `spec.penalties` is empty on every multinomial fit and the guard runs
    // zero times, while `multinomial.rs` cites this function BY NAME as the
    // protection that keeps a near-separable fit off the smoothing wall. A `for`
    // over an empty list is not an error, so nothing reported it: on penguins all
    // sixteen selected λ railed at `exp(EFFECTIVE_DF_CEILING)` to the last bit and
    // the fit collapsed to the base rate. `MultinomialFamily` is the only
    // non-default implementor of `joint_penalty_specs()`, so this hole was
    // multinomial-only and every other family's bounds are byte-identical.
    //
    // The joint specs reach the SAME bisection through the same helper; only the
    // Gram differs, because a joint penalty spans the stacked block vector rather
    // than one block's columns.
    //
    // The bound is aggregated within a DECLARED group before it is applied, and
    // that granularity is the whole difficulty. A bound computed per SPEC is
    // reference-DEPENDENT: the K per-class specs of a term are not a permutation
    // orbit in the stacked ALR basis, because the reference class's centering row
    // is `−𝟙ᵀ/K` while an active class's is `e_cᵀ − 𝟙ᵀ/K`. Relabeling therefore
    // changes each spec's pencil and its bound, and the measured fit drifts by
    // 1.637e-2 against the 1e-3 bar of
    // `multinomial_fit_is_invariant_to_reference_class_1587` (refit noise
    // exactly 0 — structural, not numerical).
    //
    // Any aggregation over a set that MAPS TO ITSELF under relabeling restores
    // invariance, but not every such set is acceptable. Aggregating over ALL
    // joint specs does restore it — and collapses every coordinate onto one
    // shared wall (measured: all λ bit-identical at `exp(10.684)`), which
    // destroys the per-class heterogeneity #1855 exists for AND makes the #1587
    // gate vacuous, since all-λ-equal is invariant trivially. That is a guard
    // silently becoming a tautology while its test goes green.
    //
    // The GROUP is the finest aggregation that is still invariant: relabeling
    // permutes a term's K per-class specs among themselves, so the minimum over
    // the group is invariant while the components (wiggliness vs null-space)
    // keep their own separate bounds. A spec that declares no group stands
    // alone — which is what every family other than the multinomial gets, since
    // `group` defaults to `None`.
    if !layout.joint_specs.is_empty() {
        if let Some(joint_gram) = stacked_block_design_gram(specs) {
            // (group, tightest bound) — group counts are tiny, so a linear scan
            // beats a map and keeps the emission order deterministic.
            let mut group_bounds: Vec<(usize, f64)> = Vec::new();
            let mut solo: Vec<(usize, f64)> = Vec::new();
            for (joint_idx, joint) in layout.joint_specs.iter().enumerate() {
                let Some(gammas) = penalty_range_gammas_from_gram(&joint_gram, &joint.matrix)
                else {
                    continue; // un-projectable geometry: keep the uniform ceiling.
                };
                let Some(rho_star) = effective_df_floor_bound(&gammas, rho_box)? else {
                    continue; // floor not enforceable inside the box.
                };
                match joint.group {
                    Some(group) => match group_bounds.iter_mut().find(|(g, _)| *g == group) {
                        Some((_, best)) => *best = best.min(rho_star),
                        None => group_bounds.push((group, rho_star)),
                    },
                    None => {
                        if let Some(&outer) = layout.joint_to_outer.get(joint_idx) {
                            solo.push((outer, rho_star));
                        }
                    }
                }
            }
            let mut apply = |outer: usize, rho_star: f64| {
                if outer < n_rho {
                    let slot = &mut upper[outer];
                    if rho_star < *slot {
                        *slot = rho_star;
                    }
                }
            };
            for (joint_idx, joint) in layout.joint_specs.iter().enumerate() {
                let Some(group) = joint.group else { continue };
                let Some(&(_, rho_star)) = group_bounds.iter().find(|(g, _)| *g == group) else {
                    continue;
                };
                let Some(&outer) = layout.joint_to_outer.get(joint_idx) else {
                    continue;
                };
                apply(outer, rho_star);
            }
            for (outer, rho_star) in solo {
                apply(outer, rho_star);
            }
        }
    }
    log::debug!(
        "[EDF-FLOOR] emitted rho upper bounds (ceiling={:.3}): {:?}",
        rho_box.ceiling(),
        upper
            .iter()
            .map(|b| (b * 1e6).round() / 1e6)
            .collect::<Vec<_>>(),
    );
    Ok(upper)
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

/// Tighten one ρ upper bound to the edf-floor crossing of a term's structural γ
/// spectrum.
///
/// Shared by the per-block and the joint (#2579) paths so both obey exactly the
/// same three enforceability guards, the same 64-step bisection, and the same
/// tightest-wins rule on a shared coordinate. Keeping one body is the point: a
/// second copy is how the per-block path and the joint path would drift.
fn tighten_effective_df_floor_bound(
    gammas: &[f64],
    rho_box: RhoBox,
    slot: &mut f64,
) -> Result<(), CustomFamilyError> {
    if let Some(rho_star) = effective_df_floor_bound(gammas, rho_box)? {
        if rho_star < *slot {
            *slot = rho_star;
        }
    }
    Ok(())
}

/// The edf-floor crossing itself, or `None` when the floor is not enforceable
/// strictly inside the box.
///
/// Split out of [`tighten_effective_df_floor_bound`] because the joint path must
/// AGGREGATE bounds across a group before applying any of them; applying each
/// spec's own bound is what breaks reference invariance (#2579).
fn effective_df_floor_bound(
    gammas: &[f64],
    rho_box: RhoBox,
) -> Result<Option<f64>, CustomFamilyError> {
    let ceiling = rho_box.ceiling();
    let lower = rho_box.lower();
    // Maximum attainable structural edf (ρ → −∞) is the number of
    // design-supported penalized directions. If it cannot reach the floor even
    // unpenalized, the floor is not enforceable for this term (a
    // single-dimension range space with the floor at its own cap), so keep the
    // uniform ceiling.
    //
    // KNOWN STRUCTURAL EXEMPTION (#2608). This test is `>`, and
    // `EFFECTIVE_DF_FLOOR` is exactly `1.0`, so a RANK-1 penalty — whose edf
    // ranges over `(0, 1]` and reaches `1` only as `λ → 0` — can never satisfy
    // it and is skipped unconditionally. That is arithmetically forced rather
    // than an oversight: demanding `edf ≥ 1` of a rank-1 term is a demand for
    // `ρ = −∞`, and manufacturing a bound there would be worse than none.
    //
    // The consequence is not small, and it is why this is written down rather
    // than left to be re-derived. The null-space half of a Marra–Wood double
    // penalty IS rank-1, so the LINEAR direction of every smooth is permanently
    // exempt from the only protection against its own collapse. On penguins
    // that exemption is what converts "over-smoothed" into "the null model" —
    // a linear softmax already scores 96.5% there, so the term that dies is the
    // one carrying almost all of the signal. No amount of work on WHICH
    // coordinates share a bound (the #2579 grouping above) touches it, because
    // this term never reaches the bisection at all.
    let edf_max = unit_weight_term_edf_at_physical_strength(gammas, 0.0);
    // #2608: a term that can reach the absolute floor is held to it, exactly as
    // before. A term that CANNOT (any rank-1 penalty, whose edf sup is 1.0 and is
    // attained only at λ = 0) is held to a fraction of what it can reach, instead
    // of being skipped entirely.
    let target = if edf_max > EFFECTIVE_DF_FLOOR {
        EFFECTIVE_DF_FLOOR
    } else {
        EFFECTIVE_DF_FLOOR_RELATIVE_FRACTION * edf_max
    };
    // Still refuse when even the relative target is unreachable — a term with no
    // design support at all (`edf_max = 0`) has nothing to retain, and bounding ρ
    // against it would manufacture a wall out of an empty range space.
    if !(edf_max > target) {
        return Ok(None);
    }
    // Bisect for ρ* with edf(ρ*) = floor on [lower, ceiling]; edf is monotone
    // decreasing in ρ. If edf at the ceiling still exceeds the floor, the
    // uniform ceiling already retains enough df — keep it.
    if unit_weight_term_edf(gammas, ceiling)? >= target {
        return Ok(None);
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
    // eigenvalues can put the edf=1 crossing just outside the `lower`
    // wall of the rho box. Evaluating at `lower` (the real floor) rather
    // than `-ceiling` guarantees the crossing bracketed below is strictly
    // inside `(lower, ceiling)`, so the emitted upper stays above `lower`.
    if unit_weight_term_edf(gammas, lower)? <= target {
        return Ok(None);
    }
    let mut lo = lower;
    let mut hi = ceiling;
    for _ in 0..64 {
        let mid = 0.5 * (lo + hi);
        if unit_weight_term_edf(gammas, mid)? >= target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let rho_star = 0.5 * (lo + hi);
    log::debug!(
        "[EDF-FLOOR] rank={} edf_max={edf_max:.6} target={target:.6} rho*={rho_star:.6} \
         box=[{lower:.3},{ceiling:.3}] gamma_min={:.6e} gamma_max={:.6e}",
        gammas.iter().filter(|g| **g > 0.0).count(),
        gammas.iter().cloned().fold(f64::INFINITY, f64::min),
        gammas.iter().cloned().fold(0.0_f64, f64::max),
    );
    // Guarding on `lower` keeps the emitted upper strictly above the box floor.
    // Callers apply tightest-wins: a coordinate must retain the floor for EVERY
    // term contributing to it.
    if rho_star > lower + 1e-6 {
        return Ok(Some(rho_star));
    }
    Ok(None)
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
/// The anchor is the MAXIMAL-smoothing ρ, the [`RhoBox`] ceiling: it is the most
/// smoothed admissible point, so every penalized term is collapsed onto its
/// penalty nullspace there, the surviving low-dimensional problem is the
/// parametric fit, and its mode is unique — which is the one property the
/// selection rule needs, since a unique mode is what the caller's coefficients
/// cannot influence.
///
/// It is deliberately NOT the per-term
/// [`effective_df_floor_rho_upper_bounds`], even though the two coincide for
/// every term the absolute df floor exempts. That bound answers a different
/// question — how much df a term must RETAIN — and #2608 made it a partial
/// collapse for rank-deficient terms. A partially collapsed term is still
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
                log::info!(
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
            log::info!(
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
    /// carried forward rather than rediscovered.
    ///
    /// The final waypoint uses `rho_target` verbatim rather than the
    /// reconstructed `ρ_A + 1·(ρ − ρ_A)`: the endpoint must be the mode *at the
    /// requested ρ* bitwise, because everything downstream binds the mode and
    /// its ρ as one identity.
    fn sweep(&self, steps: usize) -> Result<SweptEndpoint, AnchoredContinuationRefusal> {
        let mut carried: Option<ConstrainedWarmStart> = None;
        for step in 0..=steps {
            let waypoint = if step == steps {
                self.rho_target.clone()
            } else if step == 0 {
                self.rho_anchor.clone()
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
            log::info!(
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
    log::info!(
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
    let reduced_total: usize = specs.iter().map(|s| s.design.ncols()).sum();
    let joint_specs = pulled_back_joint_penalty_specs(family, &canonical, reduced_total)?;

    let label_layout = penalty_label_layout_with_joint(specs, penalty_counts.clone(), joint_specs)?;
    let mut rho0 = label_layout.initial_rho.clone();
    let (persistent_warm_start_cache, mut persistent_warm_start) =
        load_persistent_custom_family_warm_start::<F>(family, specs, options, rho0.len());
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
        .map_err(|error| CustomFamilyError::Optimization {
            context: "fit_custom_family no-smoothing inner solve",
            reason: format!("{error}; no fit was assembled"),
        })?;
        let warm_start = constrained_warm_start_from_inner(&rho0, &inner);
        store_persistent_custom_family_warm_start(
            persistent_warm_start_cache.as_ref(),
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
            geometry,
            reported_beta,
        } = posterior;
        let geometry = Some(geometry);
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
            &mut inner.block_states,
            reported_beta.as_ref(),
        )
        .map_err(|reason| CustomFamilyError::Optimization {
            context: "fit_custom_family no-smoothing reported posterior mean",
            reason: reason.to_string(),
        })?;
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
            },
        );
    }

    // Exact Hessians remain declared whenever the assembled family can supply
    // them, but #2359 reserves that order-four work for the terminal minimum
    // certificate. Search uses the exact analytic gradient. Small iteration
    // budgets still run through this same outer solver and must earn its
    // convergence certificate; they are not a production shortcut to an
    // unoptimized fit.
    use gam_problem::OuterEval;
    use gam_solve::model_types::EstimationError;
    use gam_solve::rho_optimizer::{OuterEvalOrder, OuterProblem};

    let screening_cap = Arc::new(AtomicUsize::new(0));
    let outer_inner_cap = options
        .outer_inner_max_iterations
        .clone()
        .unwrap_or_else(|| Arc::new(AtomicUsize::new(options.inner_max_cycles.max(1))));
    outer_inner_cap.store(options.inner_max_cycles.max(1), Ordering::Relaxed);
    // #2349 — shared "re-evaluate COLD" pulse. The outer cost-stall guard raises
    // it when it grants a STUCK-stall escape (a near-separating profiled fit
    // whose warm-started trajectory carries value hysteresis on a near-flat
    // inner ridge). The outer-eval closures below observe it, drop the warm
    // cache, and re-solve the inner problem cold so search descends a consistent
    // objective surface instead of grinding to `max_iter` at a non-stationary
    // point.
    let outer_force_cold = Arc::new(AtomicBool::new(false));
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
    let bfgs_step_cap = Some(FIRST_ORDER_BFGS_LOGLAMBDA_STEP_CAP);
    // EFS / HybridEfs structural property (`H^{-1/2} B_k H^{-1/2} ≽ 0` plus a
    // parameter-independent nullspace, Wood-Fasiolo) fails for multi-block
    // families whose joint likelihood Hessian depends on β.
    let multi_block_beta_dependent =
        specs.len() > 1 && family.exact_newton_joint_hessian_beta_dependent();
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
    let rho_box = RhoBox::new(
        RhoLowerWall(options.rho_lower_bound),
        RhoCeiling(EFFECTIVE_DF_CEILING),
    )?;
    // The per-coordinate ρ at which each penalized term's structural effective
    // df reaches one. This is both the optimizer's upper bound and — because
    // every term sits on its penalty nullspace there, leaving a unique
    // parametric mode — the anchor of the #2366 continuation below.
    let rho_upper_bounds =
        effective_df_floor_rho_upper_bounds(specs, &label_layout, n_rho, rho_box)?;
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
    let continuation_anchor = Array1::<f64>::from_elem(n_rho, rho_box.ceiling());

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
            log::warn!(
                "[OUTER] coefficient-objective continuation declined with typed refusal: \
                 {refusal}. The armed coefficient mode is therefore selected by the caller's \
                 seed rather than by the continuation, so it is not the #2366 canonical mode \
                 for this fit"
            );
            None
        }
    };
    let initial_warm_cache = if let Some(certified) = objective_homotopy_seed {
        log::info!(
            "[OUTER] coefficient-objective continuation certified at {} steps: endpoint \
             discrepancy {:.3e} <= inner tolerance {:.3e}; observed contraction factor {:?}",
            certified.certificate.steps,
            certified.certificate.endpoint_discrepancy,
            certified.certificate.inner_tolerance,
            certified.certificate.observed_contraction_factor,
        );
        Some(certified.warm_start)
    } else if family.exact_newton_joint_hessian_beta_dependent()
        && !family.inner_coefficient_objective_is_globally_convex()
    {
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
                log::info!(
                    "[OUTER] #2661 anchored continuation certified at {} steps: endpoint \
                     discrepancy {:.3e} <= inner tolerance {:.3e}; observed contraction \
                     factor {:?}",
                    certified.certificate.steps,
                    certified.certificate.endpoint_discrepancy,
                    certified.certificate.inner_tolerance,
                    certified.certificate.observed_contraction_factor,
                );
                Some(certified.warm_start)
            }
            Err(refusal) => {
                log::info!(
                    "[OUTER] #2661 anchored continuation declined with typed refusal: {refusal}"
                );
                persistent_warm_start.clone()
            }
        }
    } else {
        persistent_warm_start.clone()
    };
    // What "cold" means when the stall guard drops the warm cache. Dropping it
    // to `None` sends the inner solve back to whatever coefficients the caller
    // supplied — trajectory-independent, but arbitrary, and for a nonconvex
    // family that is a seed selecting a branch. The point of the pulse is to
    // re-solve on a surface that does not depend on the path taken, so the
    // fallback is the anchored mode: cold means CANONICAL, not arbitrary.
    let canonical_seed = initial_warm_cache.clone();
    let problem = OuterProblem::new(n_rho)
        .with_stuck_stall_cold_reeval_signal(Arc::clone(&outer_force_cold))
        .with_gradient(cap_gradient)
        .with_hessian(hessian)
        .with_prefer_gradient_only(true)
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
        .with_disable_fixed_point(multi_block_beta_dependent)
        .with_tolerance(options.outer_tol)
        .with_rel_cost_tolerance(options.outer_rel_cost_tol)
        .with_max_iter(options.outer_max_iter)
        .with_bfgs_step_cap(bfgs_step_cap)
        .with_seed_config(family.outer_seed_config(n_rho))
        .with_initial_rho(rho0.clone())
        .with_screen_initial_rho(options.screen_initial_rho)
        // n-scaled profiled-criterion calibration: absolute gradient floor =
        // max(outer_tol, n·1e-9). Mirrors the primary REML outer
        // (solver/estimate.rs) and the spatial exact-joint path.
        .with_objective_scale(if n_obs > 0 { Some(n_obs as f64) } else { None })
        .with_problem_size(n_obs, p_total.max(1))
        // Per-coordinate ρ box bounds. The uniform ceiling
        // [`EFFECTIVE_DF_CEILING`] keeps the optimizer out of the dead-flat
        // λ ≈ 10⁹ region where ARC's quadratic model breaks down, the retry-stall
        // detector fires, and downstream empty-block_states crashes surface —
        // while sitting ABOVE the heavy-but-finite REML optima the data
        // legitimately selects (the #1561/#2356 location-scale mean wants ρ ≈ 11;
        // the former ρ ≤ 10 cap railed it into a spurious under-smoothed optimum).
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
            Array1::<f64>::from_elem(n_rho, rho_box.lower()),
            rho_upper_bounds.clone(),
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
    // A low-level caller-keyed session wins. Otherwise derive the outer stream
    // from the same explicit store and structural key used by the block record
    // and cross-fit artifact owners.
    let cache_session = options.cache_session.clone().or_else(|| {
        persistent_warm_start_cache.as_ref().and_then(|cache| {
            gam_solve::persistent_warm_start::open_outer_session(&cache.store, &cache.key)
        })
    });
    let problem = if let Some(session) = cache_session {
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
        // #2349: consume the cold-reeval pulse once per outer evaluation. When
        // active (a near-separating warm-start-hysteresis stall the outer guard
        // flagged), every inner solve in this evaluation drops the warm cache
        // and runs cold + uncapped so search descends a trajectory-independent
        // objective surface.
        let force_cold = outer.take_force_cold();
        if force_cold {
            outer_options
                .outer_inner_max_iterations
                .as_ref()
                .map(|cap| cap.store(0, Ordering::Relaxed));
        }
        // Genuinely value-only fulfilment (#979). A `Value` request from an outer
        // cost, screening, or reactive-domain probe never consumes the outer
        // gradient. The inner solve in `EvalMode::ValueOnly` already produces the
        // converged block β; surface it as `inner_beta_hint` (and into
        // `outer.warm_cache`) with a zero-length gradient and skip the full
        // k²·n·p² coupled-joint LAML gradient assembly.
        if matches!(order, OuterEvalOrder::Value) {
            let warm_ref = if force_cold {
                canonical_seed.as_ref()
            } else {
                screened_outer_warm_start(outer.warm_cache.as_ref(), rho)
            };
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
                    let failure = if eval.inner_converged {
                        CustomFamilyError::trial_point(format!(
                            "custom-family value-only outer objective was non-finite ({})",
                            eval.objective
                        ))
                    } else {
                        inner_solve_not_converged_error(&eval.inner, rho.len(), 0)
                    };
                    outer.warm_cache = Some(eval.warm_start);
                    outer.last_error = Some(failure);
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
                    outer.last_error = Some(failure.clone());
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
        let warm_ref = if force_cold {
            canonical_seed.as_ref()
        } else {
            screened_outer_warm_start(outer.warm_cache.as_ref(), rho)
        };
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
                let failure = inner_solve_not_converged_error(&eval.inner, rho.len(), 0);
                outer.warm_cache = Some(eval.warm_start.clone());
                outer.last_error = Some(failure);
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
                // #2349: keep the cap uncapped while the cold-reeval latch is
                // active so cold solves reach their fixed point.
                if !force_cold {
                    update_custom_outer_inner_cap_from_warm_start(
                        &outer_options,
                        &warm_start,
                        Some(gradient_norm),
                        &mut outer.initial_gradient_norm,
                    );
                }
                outer.warm_cache = Some(warm_start.clone());
                store_persistent_custom_family_warm_start(
                    persistent_warm_start_cache.as_ref(),
                    specs,
                    &warm_start,
                );
                outer.last_error = None;
                eval
            }
            Ok(eval) => {
                outer.last_error = Some(CustomFamilyError::trial_point(format!(
                    "custom-family outer objective/derivatives became non-finite \
                     (objective={})",
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
                outer.last_error = Some(failure.clone());
                return Err(EstimationError::CustomFamily(failure));
            }
        };
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
        };
        log::debug!(
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

    let mut obj = problem.build_objective_with_screening_proxy(
        CustomOuterState::new_with_cold_signal(
            initial_warm_cache,
            Arc::clone(&outer_force_cold),
        )
        .with_inner_cap(
            Arc::clone(&outer_inner_cap),
            options.inner_max_cycles.max(1),
        )
        .with_outer_derivative_pilot(family.outer_derivative_pilot_schedule()),
        |outer: &mut CustomOuterState, rho: &Array1<f64>| {
            // Always use warm cache when available — the previous inner solution
            // gives a much better starting point. This was previously disabled for
            // exact-Hessian families, forcing every inner solve to start from
            // scratch (5-10 Newton steps instead of 1-2 with warm start).
            //
            // #2349: once the outer cost-stall guard has raised the cold-reeval
            // pulse (near-separating warm-start hysteresis), drop the warm cache
            // and run this probe cold + uncapped so the profiled objective is a
            // consistent function of ρ.
            let force_cold = outer.take_force_cold();
            let warm_ref = if force_cold {
                outer_options
                    .outer_inner_max_iterations
                    .as_ref()
                    .map(|cap| cap.store(0, Ordering::Relaxed));
                canonical_seed.as_ref()
            } else {
                screened_outer_warm_start(outer.warm_cache.as_ref(), rho)
            };
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
                    // #2349: while the cold-reeval latch is active, leave the cap
                    // uncapped so every cold solve reaches its fixed point.
                    if !force_cold {
                        update_custom_outer_inner_cap_from_warm_start(
                            &outer_options,
                            &eval.warm_start,
                            None,
                            &mut outer.initial_gradient_norm,
                        );
                    }
                    outer.warm_cache = Some(eval.warm_start);
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
                        inner_solve_not_converged_error(&eval.inner, rho.len(), 0)
                    };
                    outer.warm_cache = Some(eval.warm_start);
                    outer.last_error = Some(failure);
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
                    outer.last_error = Some(failure.clone());
                    Err(EstimationError::CustomFamily(failure))
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
                let failure = CustomFamilyError::UnsupportedConfiguration {
                    reason: "custom-family EFS requires an identity per-block \
                         penalty-coordinate layout with no fixed, tied, or joint penalties"
                        .to_string(),
                };
                outer.last_error = Some(failure.clone());
                return Err(EstimationError::CustomFamily(failure));
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
                Ok((eval, warm, true, _inner)) => {
                    outer.warm_cache = Some(warm);
                    outer.last_error = None;
                    Ok(eval)
                }
                Ok((_eval, warm, false, inner)) => {
                    let failure = inner_solve_not_converged_error(&inner, rho.len(), 0);
                    outer.warm_cache = Some(warm);
                    outer.last_error = Some(failure.clone());
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
                    outer.last_error = Some(failure.clone());
                    Err(EstimationError::CustomFamily(failure))
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
                    let failure = CustomFamilyError::trial_point(format!(
                        "custom-family seed-screening proxy produced non-finite score {score}"
                    ));
                    outer.warm_cache = Some(warm_start);
                    outer.last_error = Some(failure.clone());
                    // Screening RANKS seeds; it does not decide whether the
                    // problem is fittable. `rank_seeds_with_screening`
                    // propagates this `Err` verbatim, and the seed loop then
                    // asks `is_trial_point_infeasible()` -- answering `false`
                    // for `RemlOptimizationFailed` routes it into
                    // `fatal_outer_evaluation("outer seed screening")`, a hard
                    // `return Err` that ends the fit over a ranking probe. The
                    // sibling `Err(e)` arm immediately below already reports
                    // its screening failure as a trial-point refusal for
                    // exactly this reason (#2590); a non-finite score at THIS
                    // seed is the same statement about the same seed, and was
                    // the one shape left graded fatal (#2627).
                    Err(EstimationError::CustomFamily(failure))
                }
                Err(e) => {
                    // A failure to screen this seed is a statement about this
                    // seed's rho (#2590).
                    let failure = e.into_trial_point();
                    outer.last_error = Some(failure.clone());
                    Err(EstimationError::CustomFamily(failure))
                }
            }
        },
    )
    .with_seed_inner_state(|outer: &mut CustomOuterState, beta: &Array1<f64>| {
        outer.seed_cached_beta(n_rho, specs, beta)
    })
    .with_exact_polish(CustomOuterState::begin_exact_polish)
    // EFS may discover the optimum, but only the labeled analytic evaluator
    // owns the exact objective/gradient/coefficient-mode identity consumed by
    // fit assembly. Force the runner's final full-fidelity installation
    // through that evaluator regardless of the search plan.
    .with_terminal_eval_order(if need_outer_hessian {
        OuterEvalOrder::ValueGradientHessian
    } else {
        OuterEvalOrder::ValueAndGradient
    });

    let outer_result = problem.run_certified(&mut obj, "custom family");

    let last_error_detail = obj
        .state
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
            // objective evaluation, including seed-screening probes and rejected
            // trial steps, so on a failed run it holds wherever the search died.
            // Emitting it under the same name minted a second definition, and the
            // two do diverge: on the #2501 by-group seed-3 refusal both appear in
            // ONE message, agreeing on seven of eight coordinates and differing by
            // a full 18.0 on the eighth, because the last evaluation was a
            // near-floor screening probe while the best iterate sat interior.
            // `{e}` already carries the optimizer-owned checkpoint; this reports
            // the cache as what it is.
            let last_evaluated_rho = obj
                .state
                .warm_cache
                .as_ref()
                .map(|warm| warm.rho.to_vec())
                .unwrap_or_else(|| rho0.to_vec());
            if let Some(warm) = obj.state.warm_cache.as_ref() {
                store_persistent_custom_family_warm_start(
                    persistent_warm_start_cache.as_ref(),
                    specs,
                    warm,
                );
            }
            return Err(CustomFamilyError::Optimization {
                context: "fit_custom_family outer smoothing",
                reason: format!(
                    "outer smoothing optimization failed certified-fit validation after exhausting strategy fallbacks: \
                     {e}; last_evaluated_rho={last_evaluated_rho:?}; no fit was assembled.\
                     {last_error_detail}"
                ),
            });
        }
    };
    screening_cap.store(0, Ordering::Relaxed);

    // Consume the exact derivative-bearing evaluator state installed by the
    // runner's final full-fidelity synchronization. Objective, gradient,
    // smoothing coordinate, and coefficient mode form one sealed identity.
    // Warm starts are deliberately excluded: they are seeds, not evidence
    // about which nonconvex coefficient basin produced the certificate.
    let terminal = obj.state.terminal_mode.take().ok_or_else(|| {
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
    final_options.outer_inner_max_iterations = None;
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
        geometry,
        reported_beta,
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
    // a non-PD interior V_ρ yields a typed absence, never an error.
    let smoothing_corrected = match (
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
            crate::covariance::joint_smoothing_correction(
                v_cond,
                specs,
                &label_layout,
                &rho_star,
                &inner.block_states,
                outer_hessian,
                &excluded,
            )
            .map_err(|reason| CustomFamilyError::Optimization {
                context: "fit_custom_family smoothing correction",
                reason: reason.to_string(),
            })?
            .map(|(correction, active_rank)| {
                (
                    correction,
                    gam_solve::model_types::SmoothingCorrectionMethod::FirstOrderIdentifiedSubspace {
                        active_rank,
                        rho_dimension: rho_star.len(),
                    },
                )
            })
        }
        _ => None,
    };
    install_reported_posterior_mean(
        family,
        specs,
        &mut inner.block_states,
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
    assemble_custom_family_fit_result(
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
        },
    )
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
    let canonical =
        gam_identifiability::canonical::canonicalize_for_identifiability_with_operating_scalars(
            raw_specs,
            &pre_fit_coefficient_coordinates(family, raw_specs),
            pre_fit_operating_scalars(family, raw_specs)?,
        )?;
    let specs: &[ParameterBlockSpec] = &canonical.reduced_specs;
    let penalty_counts = validate_blockspecs(specs)?;
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
        geometry,
        reported_beta,
    } = posterior;
    install_reported_posterior_mean(
        family,
        specs,
        &mut inner.block_states,
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
    let (outer_iterations, outer_gradient_norm, criterion_certificate, certified_theta) =
        match provenance {
            OwnedModeProvenance::UserFixed => (0, None, None, None),
            OwnedModeProvenance::CertifiedOuter {
                selected_theta,
                outer,
            } => {
                if matches!(
                    curvature_requirement,
                    OwnedModeCurvatureRequirement::CertifiedLocalMinimum
                ) && outer.criterion_certificate().hessian_psd() != Some(true)
                {
                    return Err(CustomFamilyError::Optimization {
                        context:
                            "fit_custom_family_fixed_log_lambdas_from_owned_mode outer curvature",
                        reason: "a profiled nonconvex coefficient mode requires a positive-semidefinite analytic outer Hessian certificate"
                            .to_string(),
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
                )
            }
        };

    let CustomFamilyOwnedMode {
        objective: selected_objective,
        rho,
        hyper_values,
        mut inner,
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
    let per_block = split_log_lambdas(&rho, &penalty_counts)?;
    let canonical =
        gam_identifiability::canonical::canonicalize_for_identifiability_with_operating_scalars(
            specs,
            &pre_fit_coefficient_coordinates(family, specs),
            pre_fit_operating_scalars(family, specs)?,
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
        specs,
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
        geometry,
        reported_beta,
    } = posterior;
    install_reported_posterior_mean(
        family,
        specs,
        &mut inner.block_states,
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
            smoothing_corrected: None,
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
    let canonical =
        gam_identifiability::canonical::canonicalize_for_identifiability_with_operating_scalars(
            raw_specs,
            &pre_fit_coefficient_coordinates(family, raw_specs),
            pre_fit_operating_scalars(family, raw_specs)?,
        )?;
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

/// Exact outer-criterion evidence returned by the diagnostic evaluators.
pub struct OuterCriterionDiagnostics {
    pub objective: f64,
    pub gradient: Array1<f64>,
    pub outer_hessian: Option<Array2<f64>>,
    pub inner_converged: bool,
}

