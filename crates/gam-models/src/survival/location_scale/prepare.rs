use super::*;

#[derive(Clone)]
pub(crate) struct PreparedSurvivalLocationScaleModel {
    pub(crate) family: SurvivalLocationScaleFamily,
    pub(crate) blockspecs: Vec<ParameterBlockSpec>,
    pub(crate) time_transform: TimeIdentifiabilityTransform,
    pub(crate) threshold_fixed_cols: usize,
    pub(crate) threshold_full_ncols: usize,
    pub(crate) log_sigma_fixed_cols: usize,
    pub(crate) log_sigma_full_ncols: usize,
    pub(crate) k_time: usize,
    pub(crate) k_threshold: usize,
    pub(crate) k_log_sigma: usize,
    pub(crate) k_wiggle: usize,
}

impl PreparedSurvivalLocationScaleModel {
    /// Whether this prepared model is the fully reduced, unpenalized
    /// constant-scale PARAMETRIC AFT regime (issue #736/#735/#721).
    ///
    /// In this regime the time block has collapsed to its identifiable affine
    /// null space (`reduce_time_to_parametric` fired, so `k_time == 0`), the
    /// scale is a single constant log-σ (`k_log_sigma == 0`), the mean is rigid
    /// or a plain parametric covariate effect whose default shrinkage ridge has
    /// been dropped by `survival_reduced_parametric_aft_regime` (`k_threshold ==
    /// 0`), and there is no link-wiggle or monotone time-wiggle (`k_wiggle ==
    /// 0`, `x_link_wiggle == None`, `time_wiggle_ncols == 0`). Every block is
    /// therefore parametric and UNPENALIZED — zero smoothing parameters — so the
    /// model is a plain few-parameter AFT MLE (loglogistic / lognormal, exactly
    /// what `survreg`/`lifelines` fit, including a parametric `~ age` effect) and
    /// the REML/LAML outer search is vacuous. Such fits are routed to a direct,
    /// robust parametric MLE (`fit_parametric_aft_direct_mle`) instead of the
    /// coupled exact-joint REML optimizer, which does not converge on this tiny
    /// unpenalized likelihood.
    ///
    /// Any genuinely flexible/penalized survival LS fit — smooth scale
    /// (`noise_formula = s(...)`, log_sigma smoothing penalties), smooth mean
    /// (`threshold ~ s(z)` wiggliness penalties), a link-wiggle, or an active
    /// monotone time-wiggle — keeps at least one nonzero `k_*` (the ridge-drop
    /// predicate excludes any block carrying a `nullspace_dim > 0` smoothing
    /// penalty) and so does NOT match here, keeping the full coupled exact-joint
    /// path unchanged.
    pub(crate) fn is_reduced_parametric_aft(&self) -> bool {
        self.k_time == 0
            && self.k_threshold == 0
            && self.k_log_sigma == 0
            && self.k_wiggle == 0
            && self.family.x_link_wiggle.is_none()
            && self.family.time_wiggle_ncols == 0
    }
}

/// Whether the scale block carries no penalties — a single constant `σ`
/// (the parametric-AFT regime). This is exactly the condition under which
/// `prepare_survival_location_scale_model` pins the time-warp ρ seed AT the
/// inner ρ box bound (the affine-baseline limit). On that dead-flat,
/// statistically-unidentified time ridge the seed-screening cascade has no
/// useful signal to rank — every capped proxy fit collapses to non-finite
/// cost and the cascade escalates to its uncapped final stage, paying a full
/// inner solve per seed on the near-singular Hessian (the multi-minute
/// no-iteration-log stall, #736/#735/#721). The pinned seed is already the
/// correct optimum, so screening is pure cost here.
///
/// A genuinely flexible scale (`noise_formula = s(...)`) carries log-sigma
/// penalties, never reaches the seed-pinning branch, and keeps full
/// screening.
pub(crate) fn survival_constant_scale(spec: &SurvivalLocationScaleSpec) -> bool {
    match &spec.log_sigma_block {
        CovariateBlockKind::Static(block) => block.penalties.is_empty(),
        CovariateBlockKind::TimeVarying(block) => block.penalties.is_empty(),
    }
}

pub(crate) fn survival_blockwise_fit_options(
    spec: &SurvivalLocationScaleSpec,
) -> BlockwiseFitOptions {
    BlockwiseFitOptions {
        inner_max_cycles: spec.max_iter,
        inner_tol: spec.tol,
        outer_max_iter: BLOCKWISE_OUTER_MAX_ITER,
        outer_tol: BLOCKWISE_OUTER_TOL,
        compute_covariance: true,
        cache_session: spec.cache_session.clone(),
        cache_mirror_sessions: spec.cache_mirror_sessions.clone(),
        // Constant-scale (parametric-AFT) fits pin the time-warp ρ seed at the
        // identified affine-baseline limit; re-screening that already-correct
        // seed across the flat unidentified time ridge only stalls. Genuinely
        // flexible scale/spatial fits keep the default `true` and full screening.
        screen_initial_rho: !survival_constant_scale(spec),
        ..BlockwiseFitOptions::default()
    }
}

pub(crate) fn validate_survival_location_scale_spec(
    spec: &SurvivalLocationScaleSpec,
) -> Result<(), SurvivalLocationScaleError> {
    let n = spec.event_target.len();
    let monotone_time_wiggle_ncols = spec.timewiggle_block.as_ref().map_or(0, |w| w.ncols);
    match &spec.inverse_link {
        InverseLink::Standard(StandardLink::Log) => {
            return Err(SurvivalLocationScaleError::InvalidConfiguration {
                reason: "fit_survival_location_scale does not support Standard(Log)".to_string(),
            });
        }
        InverseLink::Standard(StandardLink::Logit)
        | InverseLink::Standard(StandardLink::Probit)
        | InverseLink::Standard(StandardLink::CLogLog)
        | InverseLink::Standard(StandardLink::LogLog)
        | InverseLink::Standard(StandardLink::Cauchit)
        | InverseLink::Standard(StandardLink::Identity)
        | InverseLink::LatentCLogLog(_)
        | InverseLink::Sas(_)
        | InverseLink::BetaLogistic(_)
        | InverseLink::Mixture(_) => {}
    }
    if n == 0 {
        return Err(SurvivalLocationScaleError::InternalInvariant {
            reason: "fit_survival_location_scale: empty dataset".to_string(),
        });
    }
    if spec.age_entry.len() != n || spec.age_exit.len() != n || spec.weights.len() != n {
        bail_dim_sls!("fit_survival_location_scale: top-level input size mismatch");
    }
    if !(spec.tol.is_finite() && spec.tol > 0.0) {
        return Err(SurvivalLocationScaleError::InvalidConfiguration {
            reason: format!("fit_survival_location_scale: invalid tol {}", spec.tol),
        });
    }
    if spec.max_iter == 0 {
        return Err(SurvivalLocationScaleError::InvalidConfiguration {
            reason: "fit_survival_location_scale: max_iter must be > 0".to_string(),
        });
    }
    if !spec.derivative_guard.is_finite() || spec.derivative_guard <= 0.0 {
        return Err(SurvivalLocationScaleError::InvalidConfiguration {
            reason: format!(
                "fit_survival_location_scale: derivative_guard must be > 0, got {}",
                spec.derivative_guard
            ),
        });
    }
    validate_time_block(
        n,
        &spec.time_block,
        spec.derivative_guard,
        monotone_time_wiggle_ncols,
    )?;
    validate_cov_block_kind("threshold_block", n, &spec.threshold_block)?;
    validate_cov_block_kind("log_sigma_block", n, &spec.log_sigma_block)?;
    if let Some(w) = spec.timewiggle_block.as_ref() {
        if w.ncols == 0 {
            return Err(SurvivalLocationScaleError::InvalidConfiguration {
                reason: "timewiggle_block must have at least one coefficient".to_string(),
            });
        }
        if w.ncols >= spec.time_block.design_exit.ncols() {
            return Err(SurvivalLocationScaleError::InvalidConfiguration {
                reason: format!(
                    "timewiggle_block.ncols must be smaller than time_block columns: wiggle={}, total={}",
                    w.ncols,
                    spec.time_block.design_exit.ncols()
                ),
            });
        }
        if w.knots.len() < 2 * (w.degree + 1) {
            return Err(SurvivalLocationScaleError::InvalidConfiguration {
                reason: format!(
                    "timewiggle_block knot vector is too short for degree {}: got {} knots",
                    w.degree,
                    w.knots.len()
                ),
            });
        }
    }
    if let Some(w) = spec.linkwiggle_block.as_ref() {
        validatewiggle_block(n, w)?;
    }
    for i in 0..n {
        if !spec.age_entry[i].is_finite()
            || !spec.age_exit[i].is_finite()
            || spec.age_exit[i] < spec.age_entry[i]
        {
            return Err(SurvivalLocationScaleError::InvalidConfiguration {
                reason: format!(
                    "fit_survival_location_scale: invalid interval at row {} (entry={}, exit={})",
                    i + 1,
                    spec.age_entry[i],
                    spec.age_exit[i]
                ),
            });
        }
        if !spec.weights[i].is_finite() || spec.weights[i] < 0.0 {
            return Err(SurvivalLocationScaleError::InvalidConfiguration {
                reason: format!(
                    "fit_survival_location_scale: invalid weight at row {} ({})",
                    i + 1,
                    spec.weights[i]
                ),
            });
        }
        if !spec.event_target[i].is_finite() || !(0.0..=1.0).contains(&spec.event_target[i]) {
            return Err(SurvivalLocationScaleError::ConstraintViolation {
                reason: format!(
                    "fit_survival_location_scale: event_target must be in [0,1], found {} at row {}",
                    spec.event_target[i],
                    i + 1
                ),
            });
        }
    }
    Ok(())
}

pub(crate) fn prepare_survival_location_scale_model(
    spec: &SurvivalLocationScaleSpec,
) -> Result<PreparedSurvivalLocationScaleModel, String> {
    validate_survival_location_scale_spec(spec)?;
    let n = spec.event_target.len();
    let protected_timewiggle_cols = spec.timewiggle_block.as_ref().map_or(0, |w| w.ncols);
    // Constant-scale AFT regime: a single global σ identifies the time baseline
    // only through its affine `1 + log t` transform (the parametric AFT), so the
    // flexible I-spline time-warp's non-affine deviation is statistically
    // unidentified (issue #736/#735/#721). When there is also no monotone
    // timewiggle reintroducing flexibility, reduce the time block to its
    // identifiable parametric (affine null-space) rank so the inner coupled
    // exact-joint solve has no unconstrained free direction to choke on. A
    // genuinely flexible scale (`noise_formula = s(...)`, log_sigma penalties
    // present) or an active timewiggle keeps the full monotone I-spline because
    // the varying σ / wiggle DOES identify the non-affine baseline shape.
    let reduce_time_to_parametric = survival_constant_scale(spec) && protected_timewiggle_cols == 0;
    // Log entry/exit times for the canonical unit-log-t warp pin (issue #892).
    // The reduced AFT warp is folded into the geometry offsets as the EXACT
    // `log t` transform built straight from the event times — `log t` value at
    // entry/exit and `1/t` derivative at exit — bypassing the I-spline's curved
    // image of log t (the residual curvature was what kept σ miscalibrated). The
    // floor matches `checked_log_survival_times` (survival_construction.rs), the
    // same map under which the I-spline time basis is built over `log t`.
    let log_time_entry = spec.age_entry.mapv(|t| {
        t.max(crate::survival::construction::SURVIVAL_TIME_FLOOR)
            .ln()
    });
    let log_time_exit = spec.age_exit.mapv(|t| {
        t.max(crate::survival::construction::SURVIVAL_TIME_FLOOR)
            .ln()
    });
    let mut time_prepared = prepare_identified_time_block(
        &spec.time_block,
        spec.derivative_guard,
        protected_timewiggle_cols,
        reduce_time_to_parametric,
        log_time_entry.view(),
        log_time_exit.view(),
    )?;

    if time_prepared.initial_beta.is_none() {
        // Use the AUGMENTED derivative offset (issue #892): on the pinned-warp
        // path the guard `(X' z_c) β_c + offset' ≥ guard` is built against the
        // folded offset, so the seed must satisfy the same offset to land
        // feasible.
        time_prepared.initial_beta = structural_time_initial_beta_guess(
            &time_prepared.design_derivative_exit,
            &time_prepared.derivative_offset_exit,
            &spec.age_exit,
            spec.derivative_guard,
            time_prepared.coefficient_lower_bounds.as_ref(),
        );
    }

    let time_solver_design = DesignMatrix::Dense(DenseDesignMatrix::from(Arc::new(
        MultiChannelOperator::new(vec![
            DesignMatrix::Dense(DenseDesignMatrix::from(shared_dense_arc(
                &time_prepared.design_entry,
            ))),
            DesignMatrix::Dense(DenseDesignMatrix::from(shared_dense_arc(
                &time_prepared.design_exit,
            ))),
            DesignMatrix::Dense(DenseDesignMatrix::from(shared_dense_arc(
                &time_prepared.design_derivative_exit,
            ))),
        ])?,
    )));
    // Augmented offsets (issue #892): on the pinned-warp reduce path these carry
    // the folded unit-log-t value/derivative contributions; on every other path
    // they equal `spec.time_block.offset_*` verbatim.
    let time_stacked_offset = gam_linalg::utils::stack_offsets(&[
        &time_prepared.offset_entry,
        &time_prepared.offset_exit,
        &time_prepared.derivative_offset_exit,
    ]);
    // Canonical n-row view of the time block: `spec.design` is the n-row
    // exit design (one row per observation, len(eta_canonical) = n).
    // The solver's stacked `[entry; exit; derivative_exit]` operator and
    // its matching `3*n`-row offset live in `spec.stacked_design` /
    // `spec.stacked_offset`; the solver consumes those via
    // `solver_design()` / `solver_offset()`.  The audit and shape policy
    // only read `spec.design`, so every block's audit-visible row count
    // is `n`.
    let time_canonical_design: DesignMatrix =
        DesignMatrix::Dense(DenseDesignMatrix::from(time_prepared.design_exit.clone()));
    let timespec = ParameterBlockSpec {
        name: "time_transform".to_string(),
        design: time_canonical_design,
        offset: time_prepared.offset_exit.clone(),
        penalties: time_prepared
            .penalties
            .iter()
            .cloned()
            .map(PenaltyMatrix::Dense)
            .collect(),
        nullspace_dims: time_prepared.nullspace_dims.clone(),
        // A caller-supplied per-penalty time seed is indexed by the ORIGINAL
        // (un-reduced) penalty set. When the constant-scale-AFT reduction
        // dropped those penalties (`time_prepared.penalties` empty / shorter
        // than the original), that seed no longer matches and is irrelevant —
        // the reduced affine block is unpenalized — so fall back to the empty
        // seed for the (zero) retained penalties.
        initial_log_lambdas: initial_log_lambdas(
            &time_prepared.penalties,
            if time_prepared.penalties.len() == spec.time_block.penalties.len() {
                spec.time_block.initial_log_lambdas.clone()
            } else {
                None
            },
        )?,
        initial_beta: time_prepared.initial_beta.clone(),
        // Canonical-gauge ownership for the location-scale joint design.
        //
        // The three coupled blocks (`time_transform`, `threshold`,
        // `log_sigma`) each contribute a constant / intercept-like direction
        // into the flat n-row joint design the pre-fit identifiability audit
        // RRQRs (`identifiability::audit::audit_identifiability`).
        // Those constant directions are mutually aliased (e.g. for a single
        // linear covariate the `time_transform[0] ~ threshold[0]` overlap is
        // ≈ 0.98), so the joint design is genuinely rank-deficient by exactly
        // the number of surplus constants. The audit can only *attribute and
        // drop* a redundant joint column to a strictly lower-priority block;
        // with the previous uniform `gauge_priority: 100` the surplus
        // direction was un-attributable and the audit escalated to
        // `fatal = true`, refusing every well-posed fit (issue #366).
        //
        // Assigning strictly descending priorities makes the surplus
        // constant deterministically attributable: `time_transform` owns the
        // shared constant (it carries the structural monotone baseline that
        // anchors the whole location-scale parameterisation), and any aliased
        // column is dropped from the lower-priority `threshold` / `log_sigma`
        // / `linkwiggle` blocks. This is the exact gauge-ownership contract
        // documented by `identifiability::canonical::
        // canonical_five_block_gauge_ownership_succeeds_with_attribution` and
        // already used by survival marginal-slope (time=200 highest).
        gauge_priority: 200,
        jacobian_callback: None,
        stacked_design: Some(time_solver_design),
        stacked_offset: Some(time_stacked_offset),
    };

    let threshold_prep = prepare_cov_block_kind(&spec.threshold_block)?;
    let threshold_full_ncols = threshold_prep.design_exit.ncols();
    let time_reduced_to_parametric = time_block_reduces_to_parametric(
        &spec.time_block.penalties,
        spec.time_block.design_exit.ncols(),
        survival_constant_scale(spec),
        protected_timewiggle_cols,
        &spec.time_block.design_exit.to_dense(),
        log_time_exit.view(),
    );
    let threshold_fixed_cols = if time_reduced_to_parametric {
        if time_prepared.pinned_free_row_constant {
            // Reduced + unit-log-t warp PINNED (issue #892): the single surviving
            // free time column `z_c` is ROW-CONSTANT — it now carries the
            // location level. Keeping the threshold intercept too would put TWO
            // constant columns into the direct-MLE joint design, making the
            // Hessian PSD and rank-deficient by 1; the damped Newton then stalls
            // along the alias and leaves the lowest-leverage coefficient (e.g. an
            // x0:x1 interaction) stuck at its cold-start 0. Drop the LEADING
            // threshold intercept(s) so the location level is owned solely by the
            // pinned time constant `z_c`. The threshold then contributes only its
            // genuine covariate slopes; finalize pads the dropped intercept slot
            // with 0 (intercept-invariant for the g-contrast and the surface
            // anchor). Use the same leading-intercept inference the flexible path
            // uses (returns 0 for an intercept-free threshold design), so this is
            // robust to designs that carry no constant column to alias.
            infer_non_intercept_start_design(&threshold_prep.design_exit, &spec.weights)?
                .min(threshold_full_ncols)
        } else {
            // Reduced but pin did NOT fire (both-columns-free fallback): the
            // reduced I-spline columns are strictly monotone in t and span no
            // constant-in-t direction, so the time block carries no location
            // intercept the gauge contract would attribute to it. Keep the
            // threshold (location) intercept here — it is the free location level
            // b0, NOT aliased with the multiplicative scale constant nor with any
            // time-warp constant (there is none) — mirroring why the constant
            // log_sigma block keeps its intercept (`log_sigma_fixed_cols = 0`).
            // Dropping it (#736) left the raw covariate column to double as both
            // level and slope, pinning the covariate to a wrong-signed value.
            0
        }
    } else {
        infer_non_intercept_start_design(&threshold_prep.design_exit, &spec.weights)?
            .min(threshold_full_ncols)
    };
    let threshold_design = design_column_tail(
        &threshold_prep.design_exit,
        threshold_fixed_cols,
        "survival location-scale threshold design",
    )?;
    let threshold_entry_design = if let Some(x_entry) = threshold_prep.design_entry.as_ref() {
        Some(design_column_tail(
            x_entry,
            threshold_fixed_cols,
            "survival location-scale threshold entry design",
        )?)
    } else {
        None
    };
    let threshold_deriv_design =
        if let Some(x_deriv) = threshold_prep.design_derivative_exit.as_ref() {
            Some(design_column_tail(
                x_deriv,
                threshold_fixed_cols,
                "survival location-scale threshold derivative design",
            )?)
        } else {
            None
        };
    let threshold_initial_log_lambdas = initial_log_lambdas(
        &threshold_prep.penalties,
        threshold_prep.initial_log_lambdas.clone(),
    )?;
    let (threshold_penalties, threshold_nullspace_dims, threshold_initial_log_lambdas) =
        drop_leading_penalty_columns(
            &threshold_prep.penalties,
            &threshold_prep.nullspace_dims,
            threshold_initial_log_lambdas,
            threshold_fixed_cols,
            threshold_full_ncols,
            "survival location-scale threshold penalties",
        )?;
    let threshold_initial_beta = drop_leading_initial_beta(
        threshold_prep.initial_beta.clone(),
        threshold_fixed_cols,
        threshold_full_ncols,
        "survival location-scale threshold",
    )?;
    // For time-varying threshold blocks, the solver consumes a stacked
    // `[exit; entry; deriv]` operator (3*n rows) via `solver_design()`;
    // the canonical `spec.design` is the n-row exit channel only — the
    // single field both audit and shape policy read.  Non-time-varying
    // threshold blocks have no stacking: `stacked_design`/`stacked_offset`
    // stay `None` and the solver reads `design` directly.
    let (threshold_stacked_design, threshold_stacked_offset) =
        if let Some(x_entry) = threshold_entry_design.as_ref() {
            let x_deriv = threshold_deriv_design.as_ref().ok_or_else(|| {
                "time-varying threshold block is missing its exit derivative design".to_string()
            })?;
            (
                Some(DesignMatrix::Dense(DenseDesignMatrix::from(Arc::new(
                    MultiChannelOperator::new(vec![
                        threshold_design.clone(),
                        x_entry.clone(),
                        x_deriv.clone(),
                    ])?,
                )))),
                Some(gam_linalg::utils::stack_offsets(&[
                    &threshold_prep.offset,
                    &threshold_prep.offset,
                    &Array1::zeros(n),
                ])),
            )
        } else {
            (None, None)
        };
    let survival_primary_design = DesignMatrix::Dense(DenseDesignMatrix::from(Arc::new(
        BlockDesignOperator::new(vec![
            DesignBlock::Dense(DenseDesignMatrix::from(shared_dense_arc(
                &time_prepared.design_exit,
            ))),
            design_block_from_matrix(threshold_design.clone()),
        ])?,
    )));

    let log_sigma_prep = prepare_cov_block_kind(&spec.log_sigma_block)?;
    let non_intercept_start =
        infer_non_intercept_start_design(&log_sigma_prep.design_exit, &spec.weights)?;
    let log_sigma_full_ncols = log_sigma_prep.design_exit.ncols();
    // The scale channel enters the survival location-scale likelihood as
    // `z = (h(t) - eta_t(x)) / exp(eta_sigma)`: `eta_sigma` is MULTIPLICATIVE,
    // not an additive predictor. The location predictor (`eta_t`) and the
    // log-scale predictor (`eta_sigma`) are SEPARATELY identifiable even when
    // they are spanned by the same covariate basis — they enter the likelihood
    // through different sufficient statistics (the standardized residual `z`
    // versus the `-log sigma` / `z²` curvature), exactly as in the Gaussian
    // location-scale model. Residualizing the scale design against the location
    // design — the former scale-deviation reparameterization — therefore
    // imposes a spurious constraint: when `s(x)` drives BOTH channels, every
    // scale column lies in the location design's span, the residual collapses to
    // ~0, and the genuine heteroscedastic signal is erased. The flat identity
    // audit then drops those zeroed columns from the reduced spec while the
    // family keeps the full-width `x_log_sigma`, tripping
    // `exact_newton_joint_gradient_evaluation`'s "joint gradient length mismatch
    // for block 2" shape check. Identifiability across the location/scale blocks
    // is instead supplied by `output_channel_assignment` (each block drives its
    // own audit channel), so the scale design is kept RAW — an identity
    // reparameterization — matching `identified_gaussian_log_sigma_design`.
    let log_sigma_fixed_cols = 0usize;
    let scale_transform = ScaleDeviationTransform::identity(
        survival_primary_design.ncols(),
        log_sigma_prep.design_exit.ncols(),
        non_intercept_start,
    );
    let log_sigma_full_design = build_scale_deviation_operator(
        survival_primary_design.clone(),
        log_sigma_prep.design_exit.clone(),
        &scale_transform,
    )?;
    let log_sigma_design = design_column_tail(
        &log_sigma_full_design,
        log_sigma_fixed_cols,
        "survival location-scale log-sigma design",
    )?;
    let log_sigma_entry_design = if let Some(x_ls_entry) = log_sigma_prep.design_entry.as_ref() {
        let full_entry = build_scale_deviation_operator(
            survival_primary_design.clone(),
            x_ls_entry.clone(),
            &scale_transform,
        )?;
        Some(design_column_tail(
            &full_entry,
            log_sigma_fixed_cols,
            "survival location-scale log-sigma entry design",
        )?)
    } else {
        None
    };
    let log_sigma_deriv_design =
        if let Some(ls_deriv) = log_sigma_prep.design_derivative_exit.as_ref() {
            Some(design_column_tail(
                ls_deriv,
                log_sigma_fixed_cols,
                "survival location-scale log-sigma derivative design",
            )?)
        } else {
            None
        };
    let log_sigma_initial_log_lambdas = initial_log_lambdas(
        &log_sigma_prep.penalties,
        log_sigma_prep.initial_log_lambdas.clone(),
    )?;
    let (log_sigma_penalties, log_sigma_nullspace_dims, log_sigma_initial_log_lambdas) =
        drop_leading_penalty_columns(
            &log_sigma_prep.penalties,
            &log_sigma_prep.nullspace_dims,
            log_sigma_initial_log_lambdas,
            log_sigma_fixed_cols,
            log_sigma_full_ncols,
            "survival location-scale log-sigma penalties",
        )?;
    let log_sigma_initial_beta = drop_leading_initial_beta(
        log_sigma_prep.initial_beta.clone(),
        log_sigma_fixed_cols,
        log_sigma_full_ncols,
        "survival location-scale log-sigma",
    )?;

    // Reduced parametric-AFT regime (issue #736/#735/#721): when the time-warp
    // has collapsed to its affine null space, there is no wiggle, and every
    // surviving location/scale penalty is a full-rank parametric ridge
    // (`nullspace_dim == 0` — e.g. the linear-term `LinearTermRidge` on `age`),
    // drop those ridges. They are NOT wiggliness penalties: a single linear
    // coefficient has nothing to smooth, so the ridge carries no smoothing
    // parameter worth a vacuous outer ρ coordinate, and its default λ would only
    // bias the parametric coefficient away from the unpenalized
    // `survreg`/`lifelines` MLE this regime must reproduce. Dropping them
    // (exactly as the reduced time block drops its projected-to-zero penalties)
    // takes `k_threshold`/`k_log_sigma` to 0, so the dispatch
    // (`is_reduced_parametric_aft`) routes the fit to the direct unpenalized
    // parametric-AFT Newton MLE with zero outer coordinates. The OUTER ρ layout
    // (`fit_survival_location_scale_terms`) applies the SAME predicate to the
    // same boot-design penalties, so the inner and outer counts stay identical.
    // Evaluate the regime predicate on the PRE-drop block penalties
    // (`threshold_prep`/`log_sigma_prep`), which are an exact copy of the boot
    // designs the OUTER layout (`fit_survival_location_scale_terms`) inspects —
    // `prepare_cov_block_kind` clones `b.penalties`/`b.nullspace_dims` straight
    // from the block built off that boot design. Reading the same source on both
    // sides guarantees the inner and outer ρ counts are computed from identical
    // penalty/null-space metadata, so they can never diverge (a divergence would
    // desynchronise `k_threshold` between the layout and the prepared model).
    let drop_parametric_ridges = survival_reduced_parametric_aft_regime(
        &spec.time_block.penalties,
        spec.time_block.design_exit.ncols(),
        survival_constant_scale(spec),
        protected_timewiggle_cols,
        &threshold_prep.nullspace_dims,
        threshold_prep.penalties.len(),
        &log_sigma_prep.nullspace_dims,
        log_sigma_prep.penalties.len(),
        spec.linkwiggle_block.is_some(),
        &spec.time_block.design_exit.to_dense(),
        log_time_exit.view(),
    );
    let (threshold_penalties, threshold_nullspace_dims, threshold_initial_log_lambdas) =
        if drop_parametric_ridges {
            (Vec::new(), Vec::new(), Array1::<f64>::zeros(0))
        } else {
            (
                threshold_penalties,
                threshold_nullspace_dims,
                threshold_initial_log_lambdas,
            )
        };
    let (log_sigma_penalties, log_sigma_nullspace_dims, log_sigma_initial_log_lambdas) =
        if drop_parametric_ridges {
            (Vec::new(), Vec::new(), Array1::<f64>::zeros(0))
        } else {
            (
                log_sigma_penalties,
                log_sigma_nullspace_dims,
                log_sigma_initial_log_lambdas,
            )
        };

    let thresholdspec = ParameterBlockSpec {
        name: "threshold".to_string(),
        design: threshold_design.clone(),
        offset: threshold_prep.offset.clone(),
        penalties: threshold_penalties.clone(),
        nullspace_dims: threshold_nullspace_dims.clone(),
        initial_log_lambdas: threshold_initial_log_lambdas,
        initial_beta: threshold_initial_beta,
        // Lower than `time_transform` (200): the location-channel covariate
        // block yields the shared constant direction to the time baseline.
        // See the canonical-gauge ownership note on the `time_transform`
        // spec above (issue #366).
        gauge_priority: 150,
        jacobian_callback: None,
        stacked_design: threshold_stacked_design,
        stacked_offset: threshold_stacked_offset,
    };

    // Same canonical-vs-stacked split as the threshold block: time-varying
    // log_sigma stacks `[exit; entry; deriv]` (3*n rows) into
    // `stacked_design`; the canonical `spec.design` is the n-row exit
    // channel only.
    let (log_sigma_stacked_design, log_sigma_stacked_offset) =
        if let Some(ref ls_entry) = log_sigma_entry_design {
            let ls_deriv = log_sigma_deriv_design.as_ref().ok_or_else(|| {
                "time-varying log-sigma block is missing its exit derivative design".to_string()
            })?;
            (
                Some(DesignMatrix::Dense(DenseDesignMatrix::from(Arc::new(
                    MultiChannelOperator::new(vec![
                        log_sigma_design.clone(),
                        ls_entry.clone(),
                        ls_deriv.clone(),
                    ])?,
                )))),
                Some(gam_linalg::utils::stack_offsets(&[
                    &log_sigma_prep.offset,
                    &log_sigma_prep.offset,
                    &Array1::zeros(n),
                ])),
            )
        } else {
            (None, None)
        };
    let log_sigmaspec = ParameterBlockSpec {
        name: "log_sigma".to_string(),
        design: log_sigma_design.clone(),
        offset: log_sigma_prep.offset.clone(),
        penalties: log_sigma_penalties.clone(),
        nullspace_dims: log_sigma_nullspace_dims.clone(),
        initial_log_lambdas: log_sigma_initial_log_lambdas,
        initial_beta: log_sigma_initial_beta,
        // Below `time_transform` (200) and `threshold` (150): the scale
        // channel yields the shared constant direction to the location
        // blocks. See the canonical-gauge ownership note on the
        // `time_transform` spec above (issue #366).
        gauge_priority: 120,
        jacobian_callback: None,
        stacked_design: log_sigma_stacked_design,
        stacked_offset: log_sigma_stacked_offset,
    };
    let wigglespec = if let Some(w) = spec.linkwiggle_block.as_ref() {
        Some(ParameterBlockSpec {
            name: "linkwiggle".to_string(),
            design: w.design.clone(),
            offset: Array1::zeros(n),
            penalties: {
                let p_wiggle = w.design.ncols();
                w.penalties
                    .iter()
                    .map(|spec| match spec {
                        gam_terms::penalty_spec::PenaltySpec::Block {
                            local, col_range, ..
                        } => PenaltyMatrix::Blockwise {
                            local: local.clone(),
                            col_range: col_range.clone(),
                            total_dim: p_wiggle,
                        },
                        gam_terms::penalty_spec::PenaltySpec::Dense(m)
                        | gam_terms::penalty_spec::PenaltySpec::DenseWithMean {
                            matrix: m, ..
                        } => PenaltyMatrix::Dense(m.clone()),
                    })
                    .collect()
            },
            nullspace_dims: w.nullspace_dims.clone(),
            initial_log_lambdas: initial_log_lambdas(&w.penalties, w.initial_log_lambdas.clone())?,
            initial_beta: w.initial_beta.clone(),
            // Lowest of the four location-scale blocks: the optional
            // link-wiggle correction yields the shared constant direction to
            // every structural block above it. See the canonical-gauge
            // ownership note on the `time_transform` spec above (issue #366).
            gauge_priority: 80,
            jacobian_callback: None,
            stacked_design: None,
            stacked_offset: None,
        })
    } else {
        None
    };

    // σ-scaled log-t AFT location baseline (issue #892). When the rank-1 reduced
    // parametric-AFT regime fired, the time warp was removed (`h ≡ 0`); the `log t`
    // baseline is instead applied here as a per-row shift of the effective
    // location predictor on the σ-scaled `q` channel: value `−log t` at entry/exit
    // and derivative `−1/t` at exit. Then `u = inv_sigma·(log t − η_t)` and the
    // event Jacobian gains `log_g = −η_ls − log t`, the `−log σ` term that
    // identifies σ. `log_time_*` already carry the `SURVIVAL_TIME_FLOOR` floor, so
    // `1/t_exit = exp(−log t_exit)` is finite and positive.
    let location_log_time = if time_prepared.location_log_time_offset {
        Some(LocationLogTimeOffset {
            value_exit: log_time_exit.mapv(|lt| -lt),
            value_entry: log_time_entry.mapv(|lt| -lt),
            deriv_exit: log_time_exit.mapv(|lt| -((-lt).exp())),
        })
    } else {
        None
    };

    let family = SurvivalLocationScaleFamily {
        n,
        y: spec.event_target.clone(),
        w: spec.weights.clone(),
        inverse_link: spec.inverse_link.clone(),
        derivative_guard: spec.derivative_guard,
        location_log_time,
        x_time_entry: Arc::new(time_prepared.design_entry.clone()),
        x_time_exit: Arc::new(time_prepared.design_exit.clone()),
        x_time_deriv: Arc::new(time_prepared.design_derivative_exit.clone()),
        time_wiggle_knots: spec.timewiggle_block.as_ref().map(|w| w.knots.clone()),
        time_wiggle_degree: spec.timewiggle_block.as_ref().map(|w| w.degree),
        time_wiggle_ncols: protected_timewiggle_cols,
        time_linear_constraints: time_prepared.linear_constraints.clone(),
        x_threshold: threshold_design,
        x_threshold_entry: threshold_entry_design,
        x_threshold_deriv: threshold_deriv_design,
        x_log_sigma: log_sigma_design,
        x_log_sigma_entry: log_sigma_entry_design,
        x_log_sigma_deriv: log_sigma_deriv_design,
        x_link_wiggle: wigglespec.as_ref().map(|s| s.design.clone()),
        wiggle_knots: spec.linkwiggle_block.as_ref().map(|w| w.knots.clone()),
        wiggle_degree: spec.linkwiggle_block.as_ref().map(|w| w.degree),
        policy: gam_runtime::resource::ResourcePolicy::default_library(),
    };

    let mut blockspecs = vec![timespec, thresholdspec, log_sigmaspec];
    if let Some(w) = wigglespec {
        blockspecs.push(w);
    }

    Ok(PreparedSurvivalLocationScaleModel {
        family,
        blockspecs,
        time_transform: time_prepared.transform,
        threshold_fixed_cols,
        threshold_full_ncols,
        log_sigma_fixed_cols,
        log_sigma_full_ncols,
        // Time-warp smoothing-parameter count. The reduced constant-scale-AFT
        // time block is unpenalized (`k_time == 0`); the flexible regime keeps
        // one ρ per time penalty. This MUST be the same value the OUTER ρ layout
        // (`fit_survival_location_scale_terms`) computes, otherwise the inner
        // blockwise λ slicing, the outer REML search, and the reduced-parametric
        // dispatch (`is_reduced_parametric_aft`) disagree on whether the time
        // block carries a ρ.
        //
        // Source it from `survival_time_rho_count` — the single source of truth
        // for that decision — evaluated on the SAME un-reduced inputs the outer
        // layout uses (`spec.time_block.penalties`, the original time width, the
        // constant-scale/timewiggle regime). Deriving it here from
        // `time_prepared.penalties.len()` instead made `k_time` depend on whether
        // the inner reduction branch inside `prepare_identified_time_block`
        // happened to fire and clear the projected-to-zero penalties; when that
        // inner collapse did not align with the regime predicate the dispatch saw
        // a stray `k_time == 1` and routed a genuinely unpenalized parametric AFT
        // (#736: constant scale, linear mean, loglogistic) down the coupled
        // exact-joint REML path it cannot certify, instead of the direct MLE
        // bypass. Tying `k_time` to `survival_time_rho_count` makes the inner and
        // outer counts provably identical (same function, same arguments) and the
        // bypass fire exactly when the regime is fully reduced (#736 #735 #721).
        k_time: survival_time_rho_count(
            &spec.time_block.penalties,
            spec.time_block.design_exit.ncols(),
            survival_constant_scale(spec),
            protected_timewiggle_cols,
            &spec.time_block.design_exit.to_dense(),
            log_time_exit.view(),
        ),
        k_threshold: threshold_penalties.len(),
        k_log_sigma: log_sigma_penalties.len(),
        k_wiggle: spec
            .linkwiggle_block
            .as_ref()
            .map_or(0, |w| w.penalties.len()),
    })
}

pub(crate) fn finalize_survival_location_scale_fit(
    prepared: &PreparedSurvivalLocationScaleModel,
    fit: &UnifiedFitResult,
) -> Result<UnifiedFitResult, String> {
    let beta_time_reduced = fit.block_states[SurvivalLocationScaleFamily::BLOCK_TIME]
        .beta
        .clone();
    // Gauge-owned affine lift (issue #892): `β_time_raw = T · θ + a`. The
    // pinned unit-log-t warp coefficient lives in `Gauge::affine_shift`; linear
    // paths carry the zero shift.
    let beta_time = prepared
        .time_transform
        .gauge
        .lift_block_betas(&[beta_time_reduced.clone()])
        .remove(0);
    let beta_threshold_active = fit.block_states[SurvivalLocationScaleFamily::BLOCK_THRESHOLD]
        .beta
        .clone();
    let beta_threshold = expand_leading_fixed_beta(
        &beta_threshold_active,
        prepared.threshold_fixed_cols,
        prepared.threshold_full_ncols,
        "survival location-scale threshold final beta",
    )?;
    let beta_log_sigma_active = fit.block_states[SurvivalLocationScaleFamily::BLOCK_LOG_SIGMA]
        .beta
        .clone();
    let beta_log_sigma = expand_leading_fixed_beta(
        &beta_log_sigma_active,
        prepared.log_sigma_fixed_cols,
        prepared.log_sigma_full_ncols,
        "survival location-scale log-sigma final beta",
    )?;
    let beta_link_wiggle = if prepared.family.x_link_wiggle.is_some() {
        Some(
            fit.block_states[SurvivalLocationScaleFamily::BLOCK_LINK_WIGGLE]
                .beta
                .clone(),
        )
    } else {
        None
    };
    let finalization_gauge = survival_location_scale_finalization_gauge(
        &prepared.time_transform.gauge,
        beta_threshold_active.len(),
        beta_threshold.len(),
        prepared.threshold_fixed_cols,
        beta_log_sigma_active.len(),
        beta_log_sigma.len(),
        prepared.log_sigma_fixed_cols,
        beta_link_wiggle.as_ref().map_or(0, |b| b.len()),
    )?;
    let lambdas = fit.lambdas.clone();
    let lambdas_time = lambdas.slice(s![0..prepared.k_time]).to_owned();
    let lambdas_threshold = lambdas
        .slice(s![prepared.k_time..prepared.k_time + prepared.k_threshold])
        .to_owned();
    let lambdas_log_sigma = lambdas
        .slice(s![prepared.k_time + prepared.k_threshold
            ..prepared.k_time
                + prepared.k_threshold
                + prepared.k_log_sigma])
        .to_owned();
    let lambdas_linkwiggle = if prepared.k_wiggle > 0 {
        Some(
            lambdas
                .slice(s![
                    prepared.k_time + prepared.k_threshold + prepared.k_log_sigma
                        ..prepared.k_time
                            + prepared.k_threshold
                            + prepared.k_log_sigma
                            + prepared.k_wiggle
                ])
                .to_owned(),
        )
    } else {
        None
    };
    let covariance_conditional = fit
        .covariance_conditional
        .as_ref()
        .map(|cov_reduced| lift_conditional_covariance(cov_reduced, &finalization_gauge))
        .transpose()?;
    let geometry = fit
        .geometry
        .as_ref()
        .map(|geom| {
            // Precision is a quadratic form on the active tangent space. It
            // cannot be pushed into a larger raw frame by the covariance
            // congruence `T H T'`: rectangular gauges have no raw inverse, and
            // that product is not a precision. Retain the exact active Hessian
            // and compose the inner and finalization sections so saved ALO can
            // pull each raw row Jacobian back as `J_active = J_raw T`.
            let coefficient_gauge = geom
                .coefficient_gauge
                .left_compose(&finalization_gauge)
                .map_err(|reason| {
                    format!(
                        "survival location-scale coefficient-gauge finalization failed: {reason}"
                    )
                })?;
            Ok::<_, String>(FitGeometry {
                coefficient_gauge,
                penalized_hessian: geom.penalized_hessian.clone(),
                working_weights: geom.working_weights.clone(),
                working_response: geom.working_response.clone(),
            })
        })
        .transpose()?;
    survival_fit_from_parts(SurvivalLocationScaleFitResultParts {
        beta_time,
        beta_threshold,
        beta_log_sigma,
        beta_link_wiggle,
        link_wiggle_knots: prepared.family.wiggle_knots.clone(),
        link_wiggle_degree: prepared.family.wiggle_degree,
        lambdas_time,
        lambdas_threshold,
        lambdas_log_sigma,
        lambdas_linkwiggle,
        log_likelihood: fit.log_likelihood,
        reml_score: fit.reml_score,
        stable_penalty_term: fit.stable_penalty_term,
        penalized_objective: fit.penalized_objective,
        used_device: false,
        outer_iterations: fit.outer_iterations,
        outer_gradient_norm: fit.outer_gradient_norm,
        outer_converged: true,
        covariance_conditional,
        geometry,
        // Per-penalty trace / effective-d.f. from the inner blockwise fit,
        // aligned 1:1 with `fit.lambdas` in block order
        // `[time, threshold, log_sigma, wiggle]` — the same order the block
        // lambdas are sliced above. Basis-invariant under the finalize gauge
        // lift, so they apply directly to the raw block coefficient counts and
        // yield the effective per-block/total EDF `tr(F)` (issue #2106).
        penalty_block_trace: fit.penalty_block_trace().to_vec(),
        edf_by_block: fit.edf_by_block().to_vec(),
    })
}

pub(crate) fn validatewiggle_block(
    n: usize,
    b: &LinkWiggleBlockInput,
) -> Result<(), SurvivalLocationScaleError> {
    if b.design.nrows() != n {
        bail_dim_sls!(
            "linkwiggle_block design row mismatch: got {}, expected {n}",
            b.design.nrows()
        );
    }
    let p = b.design.ncols();
    if p == 0 {
        return Err(SurvivalLocationScaleError::InvalidConfiguration {
            reason: "linkwiggle_block must contain at least one basis column".to_string(),
        });
    }
    if b.knots.len() < b.degree + 2 {
        return Err(SurvivalLocationScaleError::InvalidConfiguration {
            reason: format!(
                "linkwiggle_block knot vector is too short for degree {}: got {} knots",
                b.degree,
                b.knots.len()
            ),
        });
    }
    if let Some(beta0) = &b.initial_beta
        && beta0.len() != p
    {
        bail_dim_sls!(
            "linkwiggle_block initial_beta length mismatch: got {}, expected {p}",
            beta0.len()
        );
    }
    if let Some(beta0) = &b.initial_beta {
        if let Some(beta0_slice) = beta0.as_slice() {
            validate_monotone_wiggle_beta_nonnegative(
                beta0_slice,
                "linkwiggle_block initial_beta",
            )?;
        } else {
            let beta0_values = beta0.iter().copied().collect::<Vec<_>>();
            validate_monotone_wiggle_beta_nonnegative(
                &beta0_values,
                "linkwiggle_block initial_beta",
            )?;
        }
    }
    let k = b.penalties.len();
    if let Some(rho0) = &b.initial_log_lambdas
        && rho0.len() != k
    {
        bail_dim_sls!(
            "linkwiggle_block initial_log_lambdas length mismatch: got {}, expected {k}",
            rho0.len()
        );
    }
    for (idx, s) in b.penalties.iter().enumerate() {
        match s {
            gam_terms::penalty_spec::PenaltySpec::Block {
                local, col_range, ..
            } => {
                if col_range.end > p
                    || local.nrows() != col_range.len()
                    || local.ncols() != col_range.len()
                {
                    bail_dim_sls!(
                        "linkwiggle_block penalty {idx} block shape mismatch: col_range={}..{}, local={}x{}, total_dim={p}",
                        col_range.start,
                        col_range.end,
                        local.nrows(),
                        local.ncols()
                    );
                }
            }
            gam_terms::penalty_spec::PenaltySpec::Dense(m)
            | gam_terms::penalty_spec::PenaltySpec::DenseWithMean { matrix: m, .. } => {
                let (r, c) = m.dim();
                if r != p || c != p {
                    bail_dim_sls!("linkwiggle_block penalty {idx} must be {p}x{p}, got {r}x{c}");
                }
            }
        }
    }
    Ok(())
}

pub(crate) fn validate_time_block(
    n: usize,
    b: &TimeBlockInput,
    derivative_guard: f64,
    monotone_time_wiggle_ncols: usize,
) -> Result<(), SurvivalLocationScaleError> {
    if b.design_entry.nrows() != n
        || b.design_exit.nrows() != n
        || b.design_derivative_exit.nrows() != n
        || b.offset_entry.len() != n
        || b.offset_exit.len() != n
        || b.derivative_offset_exit.len() != n
    {
        bail_dim_sls!("time_block input size mismatch");
    }
    let p = b.design_exit.ncols();
    if b.design_entry.ncols() != p || b.design_derivative_exit.ncols() != p {
        bail_dim_sls!("time_block design column mismatch across entry/exit/derivative");
    }
    if !b.time_monotonicity.is_coordinate_cone() {
        return Err(SurvivalLocationScaleError::InvalidConfiguration {
            reason: format!(
                "time_block requires a coordinate-cone monotonicity strategy by construction; got {:?}",
                b.time_monotonicity
            ),
        });
    }
    structural_time_coefficient_lower_bounds_with_monotone_time_wiggle(
        &b.design_derivative_exit,
        &b.derivative_offset_exit,
        derivative_guard,
        monotone_time_wiggle_ncols,
    )?;
    if let Some(beta0) = &b.initial_beta
        && beta0.len() != p
    {
        bail_dim_sls!(
            "time_block initial_beta length mismatch: got {}, expected {p}",
            beta0.len()
        );
    }
    let k = b.penalties.len();
    if let Some(rho0) = &b.initial_log_lambdas
        && rho0.len() != k
    {
        bail_dim_sls!(
            "time_block initial_log_lambdas length mismatch: got {}, expected {k}",
            rho0.len()
        );
    }
    for (idx, s) in b.penalties.iter().enumerate() {
        let (r, c) = s.dim();
        if r != p || c != p {
            bail_dim_sls!("time_block penalty {idx} must be {p}x{p}, got {r}x{c}");
        }
    }
    Ok(())
}
