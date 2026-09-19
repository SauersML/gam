use super::*;
use crate::fit_orchestration::FitFailure;

pub(crate) fn materialize_survival<'a>(
    parsed: &ParsedFormula,
    data: &'a Dataset,
    col_map: &HashMap<String, usize>,
    config: &FitConfig,
    entry_col: Option<&str>,
    exit_col: &str,
    event_col: &str,
    interval_right_col: Option<&str>,
    structural_only: bool,
) -> Result<MaterializedModel<'a>, WorkflowError> {
    let mut inference_notes = Vec::new();

    // Extract columns. `entry_col == None` is the right-censored shorthand
    // `Surv(time, event)`: every subject enters at time zero, so we
    // synthesize a constant-zero entry vector instead of resolving a column.
    let entry_idx = entry_col
        .map(|name| resolve_role_col(col_map, name, "entry"))
        .transpose()?;
    let exit_idx = resolve_role_col(col_map, exit_col, "exit")?;
    let event_idx = resolve_role_col(col_map, event_col, "event")?;
    use rayon::iter::{IntoParallelIterator, ParallelIterator};
    let n = data.values.nrows();
    let event = data.values.column(event_idx).to_owned();
    let event_codes = Array1::from_iter(
        event
            .iter()
            .copied()
            .enumerate()
            .map(|(i, value)| crate::survival::survival_event_code_from_value(value, i))
            .collect::<Result<Vec<_>, _>>()?,
    );
    let pairs: Result<Vec<(f64, f64)>, String> = (0..n)
        .into_par_iter()
        .map(|i| {
            let entry_val = entry_idx.map_or(0.0, |idx| data.values[[i, idx]]);
            normalize_survival_time_pair(entry_val, data.values[[i, exit_idx]], i)
        })
        .collect();
    let pairs = pairs?;
    let mut age_entry = Array1::<f64>::zeros(n);
    let mut age_exit = Array1::<f64>::zeros(n);
    for (i, (e, x)) in pairs.into_iter().enumerate() {
        age_entry[i] = e;
        age_exit[i] = x;
    }

    // Interval-censored `SurvInterval(L, R, event)`: `exit_col` carried the
    // LEFT boundary `L` (resolved into `age_exit` above), and `interval_right_col`
    // carries the RIGHT boundary `R`. The kernel's interval contribution
    // `log[S(L) − S(R)]` requires a finite `R > L` per row (`event >= 0.5`) —
    // the interval mass `P(L < T ≤ R) = S(L) − S(R)` is positive only for a
    // strictly wider-than-zero bracket (`R == L` gives `log 0 = −∞`); a
    // row with `event < 0.5` is right-censored at `L` (its `R` is ignored). We
    // resolve `age_right` here so the downstream latent time stack can evaluate
    // the baseline at `R`.
    let age_right = if let Some(right_col) = interval_right_col {
        let right_idx = resolve_role_col(col_map, right_col, "interval right")?;
        let mut right = Array1::<f64>::zeros(n);
        for i in 0..n {
            let r = data.values[[i, right_idx]];
            let is_bracketed = data.values[[i, event_idx]] >= 0.5;
            if is_bracketed {
                // Require a STRICTLY positive bracket width: the kernel's interval
                // contribution is `log[S(L) − S(R)]`, which is `log 0 = −∞` at a
                // degenerate `R == L` (zero-probability bracket) and would poison
                // the whole fit's objective instead of surfacing this per-row
                // error (#2277). `R < L` is likewise rejected.
                if !(r.is_finite()) || r <= age_exit[i] {
                    return Err(WorkflowError::InvalidConfig {
                        reason: format!(
                            "SurvInterval(L, R, event) requires a finite R > L on bracketed rows (event >= 1); row {} has L={}, R={r}",
                            i + 1,
                            age_exit[i]
                        ),
                    });
                }
                right[i] = r;
            } else {
                // Right-censored row: R is unused by the likelihood. Pin it to L
                // so the (ignored) right channel stays well-defined and the
                // `age_exit <= age_right` time-basis invariant holds.
                right[i] = age_exit[i];
            }
        }
        Some(right)
    } else {
        None
    };

    // Resolve the survival likelihood at THE seam: an explicit `Some(mode)` is
    // used as-is; an unset `None` becomes the one canonical default
    // `"transformation"` here (#2301). This is the only place the survival
    // default is materialized.
    let mut survival_mode = parse_survival_likelihood_mode(config.resolved_survival_likelihood())?;
    // `linkwiggle(...)` is a flexible-link feature defined only for the
    // location-scale and marginal-slope survival models; it is meaningless under
    // the default `transformation` (Royston-Parmar) likelihood. When the user
    // adds `linkwiggle(...)` to a `Surv(...)` formula without overriding the
    // (default) `survival_likelihood='transformation'`, the formula itself
    // selects the location-scale AFT model whose link the wiggle flexes — so
    // promote rather than reject. An EXPLICIT incompatible likelihood
    // (weibull/latent/latent-binary) is still a hard error below.
    if parsed.linkwiggle.is_some() && survival_mode == SurvivalLikelihoodMode::Transformation {
        survival_mode = SurvivalLikelihoodMode::LocationScale;
    }
    // A noise formula is the log-sigma predictor, which only the location-scale
    // likelihood has. Under the default `transformation` likelihood it selects
    // that model, as `linkwiggle(...)` does. An explicit likelihood with no sigma
    // block is refused rather than fitted with the noise formula dropped.
    if config.noise_formula.is_some() {
        if survival_mode == SurvivalLikelihoodMode::Transformation {
            survival_mode = SurvivalLikelihoodMode::LocationScale;
        }
        if survival_mode != SurvivalLikelihoodMode::LocationScale {
            return Err(WorkflowError::InvalidConfig {
                reason: format!(
                    "noise_formula requires the survival location-scale likelihood; survival_likelihood='{}' has no log-sigma predictor",
                    config.resolved_survival_likelihood()
                ),
            });
        }
    }
    // `survmodel(spec=...)` names the risk the fit estimates. Every survival
    // likelihood here fits one hazard per cause, which is the net risk. A crude
    // risk combines the cause-specific hazards, so it is refused rather than
    // fitted as net.
    if let Some(spec) = parsed.survivalspec.as_ref().and_then(|s| s.spec.as_deref()) {
        let spec = spec.to_ascii_lowercase();
        if spec == "crude" {
            return Err(WorkflowError::InvalidConfig {
                reason: "survival spec 'crude' is not supported by the one-hazard fitter; use survmodel(spec=net) and compute crude risk from separate cause-specific hazards"
                    .to_string(),
            });
        }
        if spec != "net" {
            return Err(WorkflowError::InvalidConfig {
                reason: format!(
                    "unsupported survmodel(spec='{spec}'); only spec=net is accepted by the one-hazard fitter"
                ),
            });
        }
    }
    if age_right.is_some() && survival_mode != SurvivalLikelihoodMode::Latent {
        return Err(WorkflowError::InvalidConfig {
            reason: format!(
                "interval-censored SurvInterval(L, R, event) is only defined for the latent \
                 hazard-window survival likelihood (its kernel carries the log[S(L) − S(R)] \
                 interval contribution); got survival_likelihood='{}'",
                config.resolved_survival_likelihood()
            ),
        });
    }
    // Fail fast on zero effective event mass (all-censored, OR every event-coded
    // row carries zero weight) for every survival likelihood (#789B /
    // construction-time fittability split; #2276). With no row contributing a
    // target event to the WEIGHTED likelihood, the survival likelihood has no
    // event score: the hazard direction is unidentified and the inner/outer
    // solve either spins on a flat landscape (marginal-slope) or returns a
    // numerically degenerate fit (other modes). Testing the raw event codes here
    // was weight-blind — a weights column that is zero exactly on the event rows
    // passed this gate yet every kernel drops `weight <= 0` rows, so the
    // effective event score was empty and the fit failed downstream instead of
    // here. The weight column is resolved once and reused below; when no weight
    // column is supplied it is all-ones, so this reduces to the original raw
    // event-count test. This is the single chokepoint every survival fit
    // dispatcher routes through (Surv(...) responses + all FitConfig survival
    // modes), so catching it here keeps every downstream constructor —
    // `WorkingModelSurvival`, the Royston-Parmar wrapper, the marginal-slope
    // builders — free to materialize models on censored fixtures (which the
    // engine's structural unit tests rely on) without losing the user-facing
    // safety on real fits.
    let weights = resolve_weight_column(data, col_map, config.weight_column.as_deref())?;
    let weighted_event_mass: f64 = event_codes
        .iter()
        .zip(weights.iter())
        .filter(|&(&code, _)| code > 0)
        .map(|(_, &weight)| weight)
        .sum();
    if !(weighted_event_mass > 0.0) {
        let mode_label = match survival_mode {
            SurvivalLikelihoodMode::MarginalSlope => "survival marginal-slope",
            _ => "survival fit",
        };
        return Err(WorkflowError::InvalidConfig {
            reason: format!(
                "{mode_label} requires at least one target event with positive weight; every event-coded row is absent or zero-weighted, so the weighted likelihood has no event score and cannot identify the hazard"
            ),
        });
    }
    let cause_count =
        crate::survival::cause_count_from_event_codes(event_codes.view()).into_workflow_result()?;
    if cause_count > 1
        && !matches!(
            survival_mode,
            SurvivalLikelihoodMode::Transformation | SurvivalLikelihoodMode::Weibull
        )
    {
        return Err(WorkflowError::InvalidConfig {
            reason: format!(
                "cause-specific competing risks with {cause_count} causes are currently supported for survival_likelihood='transformation' and 'weibull'; got '{}'",
                config.resolved_survival_likelihood()
            ),
        }
        .into());
    }
    // Per-cause identifiability for competing risks (#2276 hardening). The total
    // weighted-event-mass gate above guarantees SOME cause has positive mass, but
    // a cause-specific hazard block for cause `c` is unidentifiable when `c` has
    // zero positive-weight events — the total gate passes while a modeled cause
    // has none. `cause_count_from_event_codes` already rejects non-contiguous
    // codes (a code-ABSENT cause), so every cause in `1..=cause_count` is
    // code-present here; this checks that each also carries positive WEIGHTED
    // event mass (kernels drop `weight <= 0` rows, so a code-present but
    // all-zero-weighted cause has an empty effective cause-`c` score and its
    // hazard is unidentified exactly like the total zero-event case).
    if cause_count > 1 {
        for cause in 1..=cause_count {
            let cause_code = cause as u8;
            let cause_mass: f64 = event_codes
                .iter()
                .zip(weights.iter())
                .filter(|&(&code, _)| code == cause_code)
                .map(|(_, &weight)| weight)
                .sum();
            if !(cause_mass > 0.0) {
                return Err(WorkflowError::InvalidConfig {
                    reason: format!(
                        "cause-specific competing risks: cause {cause} of {cause_count} has no target event with positive weight, so its cause-specific hazard block is unidentifiable; every code-{cause} row is absent or zero-weighted"
                    ),
                });
            }
        }
    }
    if parsed.linkwiggle.is_some()
        && !matches!(
            survival_mode,
            SurvivalLikelihoodMode::LocationScale | SurvivalLikelihoodMode::MarginalSlope
        )
    {
        return Err(WorkflowError::InvalidConfig {
            reason: format!(
                "linkwiggle(...) is not defined for survival_likelihood='{}'",
                config.resolved_survival_likelihood()
            ),
        }
        .into());
    }
    if parsed.linkspec.is_some()
        && matches!(
            survival_mode,
            SurvivalLikelihoodMode::Transformation
                | SurvivalLikelihoodMode::Weibull
                | SurvivalLikelihoodMode::Latent
                | SurvivalLikelihoodMode::LatentBinary
        )
    {
        return Err(WorkflowError::InvalidConfig {
            reason: format!(
                "link(...) is not implemented for survival_likelihood='{}'",
                config.resolved_survival_likelihood()
            ),
        }
        .into());
    }
    // Hoist the survival marginal-slope z-column exclusion check above the
    // time-basis / termspec construction below.  Those downstream steps fail
    // fast on small or tightly-spaced time data (e.g. an I-spline of degree 3
    // cannot be supported by a 2-row fixture), which would otherwise swallow
    // the z-column misuse error and surface a knot-count error instead.
    // Checking here keeps the user-visible error tied to the actual config
    // problem the caller can fix (rename `z` or remove the alias) rather than
    // to an unrelated basis-shape failure further downstream.
    if matches!(survival_mode, SurvivalLikelihoodMode::MarginalSlope)
        && let Some(z_column) = config.z_column.as_deref()
    {
        let slope_parsed_for_check = match config.slope_formula.as_deref() {
            Some(ls_formula) => Some(
                parse_matching_auxiliary_formula(ls_formula, &parsed.response, "slope_formula")?
                    .1,
            ),
            None => None,
        };
        let slope_ref = slope_parsed_for_check.as_ref().unwrap_or(parsed);
        validate_marginal_slope_z_column_exclusion(
            parsed,
            slope_ref,
            z_column,
            "survival marginal-slope",
            "slope_formula",
        )?;
        // Same alias hole as the Bernoulli path (gam#2432): this survival
        // entry installs `column_map_with_alias(col_map, "z", z_column)` below,
        // so a bare `z` in the main formula binds to the score even though the
        // literal-name check above cannot see it.
        validate_marginal_slope_z_alias_exclusion(
            parsed,
            col_map,
            z_column,
            "survival marginal-slope",
        )?;
    }
    let effective_timewiggle = parsed.timewiggle.clone();
    let baseline_target_raw = match survival_mode {
        SurvivalLikelihoodMode::Weibull if effective_timewiggle.is_some() => "weibull",
        SurvivalLikelihoodMode::Weibull => "linear",
        _ => &config.baseline_target,
    };
    let baseline_cfg = initial_survival_baseline_config_for_fit(
        baseline_target_raw,
        config.baseline_scale,
        config.baseline_shape,
        config.baseline_rate,
        config.baseline_makeham,
        &age_exit,
    )?;
    if matches!(
        survival_mode,
        SurvivalLikelihoodMode::Latent | SurvivalLikelihoodMode::LatentBinary
    ) && baseline_cfg.target == SurvivalBaselineTarget::Linear
    {
        return Err(
            "latent hazard-window families require a non-linear scalar baseline target; use baseline_target weibull, gompertz, or gompertz-makeham"
                .to_string()
                .into(),
        );
    }
    let time_cfg = if effective_timewiggle.is_some() {
        // Match the CLI path: the parametric baseline plus timewiggle supplies
        // the time structure, so the base time basis is disabled.
        SurvivalTimeBasisConfig::None
    } else if survival_mode == SurvivalLikelihoodMode::Weibull {
        SurvivalTimeBasisConfig::Linear
    } else {
        parse_survival_time_basis_config(
            &config.time_basis,
            config.time_degree,
            config.time_num_internal_knots,
        )?
    };
    // The one anchor rule, shared with every other front end — see
    // `resolve_survival_time_anchor_for_mode` for why the robust interior anchor
    // exists (#751/#1790) and why it must not be decided twice (#2631).
    let time_anchor = resolve_survival_time_anchor_for_mode(
        survival_mode,
        &age_entry,
        &age_exit,
        config.survival_time_anchor,
    )?;
    let exact_derivative_guard = survival_derivative_guard_for_likelihood(survival_mode);

    // Build time basis
    let mut time_build = build_survival_time_basis(
        &age_entry,
        &age_exit,
        time_cfg.clone(),
        Some(config.time_num_internal_knots),
    )?;
    if survival_mode != SurvivalLikelihoodMode::Weibull && effective_timewiggle.is_none() {
        require_structural_survival_time_basis(&time_build.basisname, "workflow survival fitting")?;
    }
    let resolved_time_cfg = resolved_survival_time_basis_config_from_build(
        &time_build.basisname,
        time_build.degree,
        time_build.knots.as_ref(),
        time_build.keep_cols.as_ref(),
    )?;
    let time_anchor_row = evaluate_survival_time_basis_row(time_anchor, &resolved_time_cfg)?;
    center_survival_time_designs_at_anchor(
        &mut time_build.x_entry_time,
        &mut time_build.x_exit_time,
        &time_anchor_row,
    )?;
    // Interval-censored data needs the SAME monotone time basis evaluated at the
    // RIGHT boundary `R` (so `q_right = X_time(R)·β_time + offset_right`). Rebuild
    // it from the FROZEN knots (`resolved_time_cfg`, carrying the knot vector the
    // exit basis just inferred) at `age_right` in the exit slot — no knot drift —
    // and anchor-center its exit design identically. The resulting `x_exit_time`
    // row is exactly the design at `R`. `time_build_right.x_entry_time` /
    // `x_derivative_time` are unused by the interval-right channel.
    let time_build_right = if let Some(age_right) = age_right.as_ref() {
        let mut build_right = build_survival_time_basis(
            &age_entry,
            age_right,
            resolved_time_cfg.clone(),
            Some(config.time_num_internal_knots),
        )?;
        center_survival_time_designs_at_anchor(
            &mut build_right.x_entry_time,
            &mut build_right.x_exit_time,
            &time_anchor_row,
        )?;
        Some(build_right)
    } else {
        None
    };
    if effective_timewiggle.is_some() && baseline_cfg.target == SurvivalBaselineTarget::Linear {
        return Err(
            "timewiggle requires a non-linear scalar survival baseline target; \
             use baseline_target weibull, gompertz, or gompertz-makeham"
                .to_string()
                .into(),
        );
    }

    // CTN composition supplies its generated score before materialization.
    let marginal_slope_aliased_col_map = if survival_mode == SurvivalLikelihoodMode::MarginalSlope {
        match config.z_column.as_deref() {
            Some(z_column) => Some(column_map_with_alias(col_map, "z", z_column)),
            None => {
                return Err(WorkflowError::InvalidConfig {
                    reason: "marginal-slope survival materialization requires z_column"
                        .to_string(),
                });
            }
        }
    } else {
        None
    };
    let termspec_col_map = marginal_slope_aliased_col_map.as_ref().unwrap_or(col_map);
    let mut termspec = build_termspec_with_geometry_and_overrides(
        &parsed.terms,
        data,
        termspec_col_map,
        &mut inference_notes,
        config.scale_dimensions,
        config.smooth_overrides.as_ref(),
        None,
    )?;
    let mut unidentified_scalar_terms = Vec::new();
    if survival_mode == SurvivalLikelihoodMode::MarginalSlope {
        unidentified_scalar_terms.extend(prune_unidentified_linear_terms_for_marginal_slope(
            &mut termspec,
            data,
            "survival marginal-slope marginal formula",
            &mut inference_notes,
        )?);
    }

    // `survmodel(distribution=...)` in the formula names the residual law, as
    // `survival_distribution` does in the configuration, and the formula's
    // `link(...)` with its initialization options names the inverse link, as
    // `link` does; the formula wins in both. A fit without a link takes its
    // inverse link from the residual law.
    let formula_link = parsed.linkspec.as_ref();
    let link_name = formula_link
        .map(|spec| spec.link.as_str())
        .or(config.link.as_deref());
    let survival_inverse_link = crate::survival::construction::parse_survival_inverse_link(
        crate::survival::construction::SurvivalInverseLinkInput {
            link: link_name,
            mixture_rho: formula_link.and_then(|spec| spec.mixture_rho.as_deref()),
            sas_init: formula_link.and_then(|spec| spec.sas_init.as_deref()),
            beta_logistic_init: formula_link.and_then(|spec| spec.beta_logistic_init.as_deref()),
            survival_distribution: parsed
                .survivalspec
                .as_ref()
                .and_then(|s| s.survival_distribution.as_deref())
                .unwrap_or(config.survival_distribution.as_str()),
        },
    )?;
    // `loglog` and `cauchit` are single-component mixtures, not link choices a
    // link deviation can flex.
    let link_choice = if link_name.is_some_and(|name| {
        let name = name.trim();
        name.eq_ignore_ascii_case("loglog") || name.eq_ignore_ascii_case("cauchit")
    }) {
        None
    } else {
        parse_link_choice(link_name, config.flexible_link)?
    };
    // Only the location-scale likelihood fits the anchored link deviation a
    // `flexible(...)` link asks for; another likelihood would drop it.
    if link_choice.as_ref().is_some_and(|choice| {
        matches!(choice.mode, gam_terms::inference::formula_dsl::LinkMode::Flexible)
    }) && survival_mode != SurvivalLikelihoodMode::LocationScale
    {
        return Err(WorkflowError::InvalidConfig {
            reason: format!(
                "survival flexible(...) links are supported only with survival_likelihood='location-scale'; got '{}'",
                config.resolved_survival_likelihood()
            ),
        });
    }
    let effective_linkwiggle =
        effectivelinkwiggle_formulaspec(parsed.linkwiggle.as_ref(), link_choice.as_ref());
    let effective_linkwiggle_cfg = effective_linkwiggle.clone().map(|cfg| LinkWiggleConfig {
        degree: cfg.degree,
        num_internal_knots: cfg.num_internal_knots,
        penalty_orders: cfg.penalty_orders,
        double_penalty: cfg.double_penalty,
    });

    // `weights` was resolved once above for the fittability gate (#2276); reuse it.
    let threshold_offset = resolve_offset_column(data, col_map, config.offset_column.as_deref())?;
    let log_sigma_offset =
        resolve_offset_column(data, col_map, config.noise_offset_column.as_deref())?;
    let threshold_template = if let Some(k) = config.threshold_time_k {
        build_time_varying_survival_covariate_template(
            &age_entry,
            &age_exit,
            k,
            config.threshold_time_degree,
            "threshold",
        )?
    } else {
        SurvivalCovariateTermBlockTemplate::Static
    };
    let log_sigma_template = if let Some(k) = config.sigma_time_k {
        build_time_varying_survival_covariate_template(
            &age_entry,
            &age_exit,
            k,
            config.sigma_time_degree,
            "sigma",
        )?
    } else {
        SurvivalCovariateTermBlockTemplate::Static
    };
    // The slope time margin (gam#2765, gam#2767). Built from the same
    // primitive the threshold and sigma margins use, so the three blocks share
    // one knot rule, one degree admission check and one replay path.
    let slope_template = if let Some(k) = config.slope_time_k {
        if survival_mode != SurvivalLikelihoodMode::MarginalSlope {
            return Err(WorkflowError::InvalidConfig {
                reason: "slope_time_k applies to the survival marginal-slope likelihood; \
                         the slope block does not exist in the other survival modes"
                    .to_string(),
            });
        }
        build_time_varying_survival_covariate_template(
            &age_entry,
            &age_exit,
            k,
            config.slope_time_degree,
            "slope",
        )?
    } else {
        SurvivalCovariateTermBlockTemplate::Static
    };
    let log_sigmaspec = if let Some(noise) = config.noise_formula.as_deref() {
        let mut noise_parsed = parse_formula(&format!("{} ~ {noise}", parsed.response))?;
        apply_secondary_predictor_basis_parsimony(&mut noise_parsed.terms, data.values.nrows());
        // Use the same aliased col_map as the main termspec — survival
        // marginal-slope reserves `z` as a placeholder for `--z-column`,
        // and the slope/noise formula may reference it too.
        build_termspec_with_geometry_and_overrides(
            &noise_parsed.terms,
            data,
            termspec_col_map,
            &mut inference_notes,
            config.scale_dimensions,
            config.smooth_overrides.as_ref(),
            None,
        )?
    } else {
        // No `noise_formula` ⇒ default to an empty log-σ spec for every
        // survival likelihood (constant log-σ baseline owned by the family
        // adapter). The previous `LocationScale`-only branch cloned the
        // mean `termspec` here, which duplicated every threshold term onto
        // the log-σ block. For a smooth `s(x)` on the mean that was
        // structurally fatal: the canonical-gauge identifiability audit
        // saw the log-σ block as exact-aliased to threshold and (per the
        // descending priorities time=200 > threshold=150 > log_sigma=120,
        // issue #366) attributed/dropped every log-σ column, leaving the
        // solver's `ParameterBlockSpec` design at width 0 while the
        // family kept the un-audited `x_log_sigma` at the smooth's width.
        // `SurvivalLocationScaleFamily::exact_newton_joint_gradient_evaluation`
        // then errored "joint gradient length mismatch for block 2: got
        // <smooth width>, expected 0" on every REML startup seed (#512).
        // The empty default routes through the same
        // `infer_non_intercept_start_design`/`design_column_tail`
        // contract every other mode uses (yielding a 0-column
        // `x_log_sigma` that matches the spec), so the family and spec
        // agree by construction.
        TermCollectionSpec {
            linear_terms: vec![],
            random_effect_terms: vec![],
            smooth_terms: vec![],
        }
    };
    // Both supplied and CTN-generated scores have an explicit column here.
    let marginal_z_column_name = if survival_mode == SurvivalLikelihoodMode::MarginalSlope {
        match config.z_column.as_deref() {
            Some(name) => Some(name),
            None => {
                return Err(WorkflowError::InvalidConfig {
                    reason: "marginal-slope survival materialization requires z_column"
                        .to_string(),
                });
            }
        }
    } else {
        None
    };
    let (
        marginal_z,
        marginal_slopespec,
        marginal_slopespecs,
        marginal_slope_deviation_routing,
        marginal_slope_base_link,
    ) = if survival_mode == SurvivalLikelihoodMode::MarginalSlope {
        let base_link = super::marginal_slope::resolve_marginal_slope_base_link(
            parsed.linkspec.as_ref(),
            "survival marginal-slope",
        )?;
        if let Some(ls_formula) = config.slope_formula.as_deref() {
            let default_z_column = marginal_z_column_name.expect("z column present when no recipe");
            let (_, ls_parsed) =
                parse_matching_auxiliary_formula(ls_formula, &parsed.response, "slope_formula")?;
            if ls_parsed.linkspec.is_some() {
                return Err(
                        "link(...) is not supported in slope_formula for the survival marginal-slope family"
                            .to_string()
                            .into(),
                    );
            }
            if ls_parsed.timewiggle.is_some() {
                return Err(
                        "timewiggle(...) is not supported in slope_formula for the survival marginal-slope family"
                            .to_string()
                            .into(),
                    );
            }
            if ls_parsed.survivalspec.is_some() {
                return Err(
                        "survmodel(...) is not supported in slope_formula for the survival marginal-slope family"
                            .to_string()
                            .into(),
                    );
            }
            validate_marginal_slope_z_column_exclusion(
                parsed,
                &ls_parsed,
                default_z_column,
                "survival marginal-slope",
                "slope_formula",
            )?;
            let surfaces = marginal_slope_surfaces(&ls_parsed, default_z_column)?;
            let mut z = Array2::<f64>::zeros((data.values.nrows(), surfaces.len()));
            let mut specs = Vec::with_capacity(surfaces.len());
            for (surface_idx, surface) in surfaces.iter().enumerate() {
                let z_idx = resolve_role_col(col_map, &surface.z_column, "z")?;
                z.column_mut(surface_idx).assign(&data.values.column(z_idx));
                let aliased_col_map = column_map_with_alias(col_map, "z", &surface.z_column);
                let mut spec = build_termspec_with_geometry_and_overrides(
                    &surface.terms,
                    data,
                    &aliased_col_map,
                    &mut inference_notes,
                    config.scale_dimensions,
                    config.smooth_overrides.as_ref(),
                    None,
                )?;
                unidentified_scalar_terms.extend(prune_unidentified_linear_terms_for_marginal_slope(
                    &mut spec,
                    data,
                    "survival marginal-slope slope_formula",
                    &mut inference_notes,
                )?);
                specs.push(spec);
            }
            (
                Some(z),
                specs.first().cloned(),
                Some(specs),
                route_marginal_slope_deviation_blocks(
                    parsed.linkwiggle.as_ref(),
                    ls_parsed.linkwiggle.as_ref(),
                )?,
                Some(base_link),
            )
        } else {
            let default_z_column = marginal_z_column_name.expect("z column present when no recipe");
            validate_marginal_slope_z_column_exclusion(
                parsed,
                parsed,
                default_z_column,
                "survival marginal-slope",
                "slope_formula",
            )?;
            let z_idx = resolve_role_col(col_map, default_z_column, "z")?;
            let z = data.values.column(z_idx).to_owned().insert_axis(Axis(1));
            (
                Some(z),
                Some(termspec.clone()),
                Some(vec![termspec.clone()]),
                route_marginal_slope_deviation_blocks(parsed.linkwiggle.as_ref(), None)?,
                Some(base_link),
            )
        }
    } else {
        (
            None,
            None,
            None,
            MarginalSlopeDeviationRouting {
                score_warp: None,
                link_dev: None,
            },
            None,
        )
    };
    let marginal_slope_score_warp = marginal_slope_deviation_routing.score_warp;
    let marginal_slope_link_dev = marginal_slope_deviation_routing.link_dev;

    if survival_mode == SurvivalLikelihoodMode::MarginalSlope {
        if parsed.linkwiggle.is_some() {
            inference_notes.push(
                "survival marginal-slope routes formula-level linkwiggle(...) into its anchored internal link-deviation block while keeping the probit survival base link".to_string(),
            );
        }
        if marginal_slope_score_warp.is_some() {
            inference_notes.push(
                "survival marginal-slope routes slope_formula linkwiggle(...) into its anchored internal score-warp block while keeping the probit survival base link".to_string(),
            );
        }
        if marginal_slope_link_dev.is_none() && marginal_slope_score_warp.is_none() {
            inference_notes.push(
                "survival marginal-slope rigid mode is algebraic closed-form exact".to_string(),
            );
        } else {
            inference_notes.push(
                "survival marginal-slope flexible score/link mode uses calibrated de-nested cubic transport cells with analytic value evaluation and calibrated survival normalization"
                    .to_string(),
            );
        }
    }
    let marginal_slope_frailty = if survival_mode == SurvivalLikelihoodMode::MarginalSlope {
        config.frailty.validate_for_marginal_slope()?;
        Some(config.frailty.clone())
    } else {
        None
    };
    if config.frailty.is_active()
        && matches!(
            survival_mode,
            SurvivalLikelihoodMode::Transformation | SurvivalLikelihoodMode::Weibull
        )
    {
        return Err(WorkflowError::InvalidConfig {
            reason: "frailty is not supported for transformation/weibull survival models"
                .to_string(),
        }
        .into());
    }
    if config.frailty.is_active() && survival_mode == SurvivalLikelihoodMode::LocationScale {
        return Err(WorkflowError::InvalidConfig {
            reason: "config.frailty is not implemented for survival-likelihood=location-scale"
                .to_string(),
        }
        .into());
    }
    if effective_timewiggle.is_some()
        && matches!(
            survival_mode,
            SurvivalLikelihoodMode::Latent | SurvivalLikelihoodMode::LatentBinary
        )
    {
        return Err(WorkflowError::InvalidConfig {
            reason: "timewiggle is not implemented for latent survival/binary likelihoods"
                .to_string(),
        }
        .into());
    }
    let latent_loading = if matches!(
        survival_mode,
        SurvivalLikelihoodMode::Latent | SurvivalLikelihoodMode::LatentBinary
    ) {
        let frailty = &config.frailty;
        Some(latent_hazard_loading(
            frailty,
            "workflow latent survival/binary",
        )?)
    } else {
        None
    };

    // SMS owns baseline theta inside its one joint LAML chart. Freeze the
    // complete prepared time stack exactly once so a baseline probe can move
    // only the three offset channels, never designs, knots, penalties, or the
    // feasibility cone.
    let marginal_slope_time_state = if survival_mode == SurvivalLikelihoodMode::MarginalSlope {
        let prepared = prepare_survival_time_stack(
            &age_entry,
            &age_exit,
            &baseline_cfg,
            SurvivalLikelihoodMode::MarginalSlope,
            None,
            time_anchor,
            exact_derivative_guard,
            &time_build,
            effective_timewiggle.as_ref(),
            None,
        )?;
        let baseline_hyper = match baseline_cfg.target {
            SurvivalBaselineTarget::Linear => SurvivalMarginalSlopeBaselineHyperSpec::Linear {
                config: baseline_cfg.clone(),
            },
            _ => SurvivalMarginalSlopeBaselineHyperSpec::Nonlinear {
                chart: SurvivalMarginalSlopeFrozenOffsetChart::new(
                    &age_entry,
                    &age_exit,
                    &baseline_cfg,
                    &prepared.eta_offset_entry,
                    &prepared.eta_offset_exit,
                    &prepared.derivative_offset_exit,
                )?,
            },
        };
        Some((prepared, baseline_hyper))
    } else {
        None
    };

    let build_time_block = |candidate: &crate::survival::construction::SurvivalBaselineConfig| {
        let prepared = prepare_survival_time_stack(
            &age_entry,
            &age_exit,
            candidate,
            survival_mode,
            (survival_mode == SurvivalLikelihoodMode::LocationScale)
                .then_some(&survival_inverse_link),
            time_anchor,
            exact_derivative_guard,
            &time_build,
            effective_timewiggle.as_ref(),
            None,
        )?;
        let time_p = prepared.time_design_exit.ncols();
        let time_initial_log_lambdas = prepared.time_initial_log_lambdas.clone();
        let initial_beta = if survival_mode == SurvivalLikelihoodMode::LocationScale {
            None
        } else {
            Some(Array1::from_elem(time_p, 1e-4))
        };
        let time_block = TimeBlockInput {
            design_entry: prepared.time_design_entry.clone(),
            design_exit: prepared.time_design_exit.clone(),
            design_derivative_exit: prepared.time_design_derivative_exit.clone(),
            offset_entry: prepared.eta_offset_entry.clone(),
            offset_exit: prepared.eta_offset_exit.clone(),
            derivative_offset_exit: prepared.derivative_offset_exit.clone(),
            time_monotonicity:
                crate::survival::location_scale::TimeBlockMonotonicity::EnforcedByCoordinateCone,
            penalties: prepared.time_penalties.clone(),
            nullspace_dims: prepared.time_nullspace_dims.clone(),
            initial_log_lambdas: time_initial_log_lambdas,
            initial_beta,
        };
        Ok::<_, String>((prepared, time_block))
    };

    // Warm-start cache for the outer baseline-config optimization: each probe
    // runs a complete inner BFGS over ρ (log-smoothing) starting from zeros if cold; by
    // capturing the previous probe's converged ρ (threshold + log_sigma blocks) and
    // injecting it here, the next inner BFGS typically converges in 1-3 iterations
    // instead of ~10, cutting per-probe cost roughly 5-10× across the probes per fit.
    let location_scale_smoothing_warm_start: RefCell<Option<(Array1<f64>, Array1<f64>)>> =
        RefCell::new(None);
    let build_location_scale_request =
        |candidate: &crate::survival::construction::SurvivalBaselineConfig| {
            let (prepared, time_block) = build_time_block(candidate)?;
            let (initial_threshold_log_lambdas, initial_log_sigma_log_lambdas) =
                match location_scale_smoothing_warm_start.borrow().as_ref() {
                    Some((thr, lsg)) => (Some(thr.clone()), Some(lsg.clone())),
                    None => (None, None),
                };
            let spec = SurvivalLocationScaleTermSpec {
                age_entry: age_entry.clone(),
                age_exit: age_exit.clone(),
                event_target: event.clone(),
                weights: weights.clone(),
                inverse_link: survival_inverse_link.clone(),
                derivative_guard: exact_derivative_guard,
                time_block,
                thresholdspec: termspec.clone(),
                log_sigmaspec: log_sigmaspec.clone(),
                threshold_offset: threshold_offset.clone(),
                log_sigma_offset: log_sigma_offset.clone(),
                threshold_template: threshold_template.clone(),
                log_sigma_template: log_sigma_template.clone(),
                timewiggle_block: prepared.timewiggle_block,
                linkwiggle_block: None,
                initial_threshold_log_lambdas,
                initial_log_sigma_log_lambdas,
                cache_session: None,
                persistent_warm_start_store: config.persistent_warm_start_store.clone(),
                cache_mirror_sessions: Vec::new(),
            };
            Ok::<_, String>(SurvivalLocationScaleFitRequest {
                data: data.values.view(),
                spec,
                wiggle: effective_linkwiggle_cfg.clone(),
                kappa_options: config.spatial_optimization.clone(),
            })
        };

    let build_marginal_slope_request = || {
        let (prepared, baseline_hyper) = marginal_slope_time_state.as_ref().ok_or_else(|| {
            "internal error: frozen marginal-slope time state is missing".to_string()
        })?;
        let time_p = prepared.time_design_exit.ncols();
        let time_initial_log_lambdas = prepared.time_initial_log_lambdas.clone();
        let time_block = TimeBlockInput {
            design_entry: prepared.time_design_entry.clone(),
            design_exit: prepared.time_design_exit.clone(),
            design_derivative_exit: prepared.time_design_derivative_exit.clone(),
            offset_entry: prepared.eta_offset_entry.clone(),
            offset_exit: prepared.eta_offset_exit.clone(),
            derivative_offset_exit: prepared.derivative_offset_exit.clone(),
            time_monotonicity:
                crate::survival::location_scale::TimeBlockMonotonicity::StructuralISpline,
            penalties: prepared.time_penalties.clone(),
            nullspace_dims: prepared.time_nullspace_dims.clone(),
            initial_log_lambdas: time_initial_log_lambdas,
            initial_beta: Some(Array1::zeros(time_p)),
        };
        Ok::<_, String>(SurvivalMarginalSlopeFitRequest {
            data: data.values.view(),
            spec: SurvivalMarginalSlopeTermSpec {
                age_entry: age_entry.clone(),
                age_exit: age_exit.clone(),
                event_target: event.clone(),
                weights: weights.clone(),
                z: marginal_z.clone().ok_or_else(|| {
                    "marginal-slope survival requires z_column in FitConfig".to_string()
                })?,
                base_link: marginal_slope_base_link.clone().ok_or_else(|| {
                    "internal error: marginal-slope base link validation missing".to_string()
                })?,
                marginalspec: termspec.clone(),
                marginal_offset: threshold_offset.clone(),
                frailty: marginal_slope_frailty.clone().ok_or_else(|| {
                    "internal error: marginal-slope frailty validation missing".to_string()
                })?,
                derivative_guard: exact_derivative_guard,
                baseline_hyper: baseline_hyper.clone(),
                time_block,
                timewiggle_block: prepared.timewiggle_block.clone(),
                slope_template: slope_template.clone(),
                slopespec: marginal_slopespec.clone().ok_or_else(|| {
                    "marginal-slope survival is missing slope spec".to_string()
                })?,
                slopespecs: marginal_slopespecs.clone(),
                slope_offset: log_sigma_offset.clone(),
                score_warp: marginal_slope_score_warp.clone(),
                link_dev: marginal_slope_link_dev.clone(),
                latent_z_policy: config.marginal_slope_latent_policy(),
                declared_latent_law: config.declared_latent_law_grid()?,
                score_influence_jacobian: None,
            },
            options: blockwise_fit_options(config),
            kappa_options: config.spatial_optimization.clone(),
        })
    };

    let build_latent_survival_request =
        |candidate: &crate::survival::construction::SurvivalBaselineConfig| {
            let loading = latent_loading.ok_or_else(|| {
                "internal error: latent survival loading missing after frailty validation"
                    .to_string()
            })?;
            let prepared = prepare_survival_time_stack(
                &age_entry,
                &age_exit,
                candidate,
                survival_mode,
                None,
                time_anchor,
                exact_derivative_guard,
                &time_build,
                None,
                Some(loading),
            )?;
            // Interval-censored: build the matching time stack at the RIGHT
            // boundary `R` (the exit slot holds `age_right`, evaluated through the
            // frozen-knot `time_build_right`). Its exit channel is exactly the
            // `R`-evaluated design / offset / unloaded mass, which feed the
            // dedicated `_right` spec fields the kernel consumes for
            // `log[S(L) − S(R)]`. The `event_target` then marks bracketed rows
            // (`event >= 1`) with the `LATENT_SURVIVAL_EVENT_INTERVAL` sentinel
            // and leaves `event < 1` rows right-censored at `L`.
            let (time_design_right, time_offset_right, unloaded_mass_right, event_target) =
                if let (Some(age_right), Some(time_build_right)) =
                    (age_right.as_ref(), time_build_right.as_ref())
                {
                    let prepared_right = prepare_survival_time_stack(
                        &age_entry,
                        age_right,
                        candidate,
                        survival_mode,
                        None,
                        time_anchor,
                        exact_derivative_guard,
                        time_build_right,
                        None,
                        Some(loading),
                    )?;
                    if prepared_right.time_design_exit.ncols() != prepared.time_design_exit.ncols()
                    {
                        return Err(format!(
                            "interval-censored right time design has {} columns but the left/exit design has {}; the right boundary basis must share the exit basis columns",
                            prepared_right.time_design_exit.ncols(),
                            prepared.time_design_exit.ncols()
                        ));
                    }
                    let event_target = event.mapv(|v| {
                        if v >= 0.5 {
                            crate::survival::latent::LATENT_SURVIVAL_EVENT_INTERVAL
                        } else {
                            0
                        }
                    });
                    (
                        Some(prepared_right.time_design_exit.clone()),
                        Some(prepared_right.eta_offset_exit.clone()),
                        prepared_right.unloaded_mass_exit.clone(),
                        event_target,
                    )
                } else {
                    (
                        None,
                        None,
                        Array1::zeros(0),
                        event.mapv(|v| if v >= 0.5 { 1 } else { 0 }),
                    )
                };
            let time_p = prepared.time_design_exit.ncols();
            let time_initial_log_lambdas = prepared.time_initial_log_lambdas.clone();
            let time_block = TimeBlockInput {
                design_entry: prepared.time_design_entry.clone(),
                design_exit: prepared.time_design_exit.clone(),
                design_derivative_exit: prepared.time_design_derivative_exit.clone(),
                offset_entry: prepared.eta_offset_entry.clone(),
                offset_exit: prepared.eta_offset_exit.clone(),
                derivative_offset_exit: prepared.derivative_offset_exit.clone(),
                time_monotonicity:
                    crate::survival::location_scale::TimeBlockMonotonicity::EnforcedByCoordinateCone,
                penalties: prepared.time_penalties.clone(),
                nullspace_dims: prepared.time_nullspace_dims.clone(),
                initial_log_lambdas: time_initial_log_lambdas,
                initial_beta: Some(Array1::from_elem(time_p, 1e-4)),
            };
            Ok::<_, String>(LatentSurvivalFitRequest {
                data: data.values.view(),
                spec: LatentSurvivalTermSpec {
                    age_entry: age_entry.clone(),
                    age_exit: age_exit.clone(),
                    event_target,
                    weights: weights.clone(),
                    derivative_guard: exact_derivative_guard,
                    time_block,
                    time_design_right,
                    age_right: if time_offset_right.is_some() {
                        age_right.clone()
                    } else {
                        None
                    },
                    time_offset_right,
                    unloaded_mass_entry: prepared.unloaded_mass_entry,
                    unloaded_mass_exit: prepared.unloaded_mass_exit,
                    unloaded_mass_right,
                    unloaded_hazard_exit: prepared.unloaded_hazard_exit,
                    meanspec: termspec.clone(),
                    mean_offset: threshold_offset.clone(),
                    baseline_config: candidate.clone(),
                },
                frailty: config.frailty.clone(),
                options: blockwise_fit_options(config),
            })
        };

    let build_latent_binary_request =
        |candidate: &crate::survival::construction::SurvivalBaselineConfig| {
            let loading = latent_loading.ok_or_else(|| {
                "internal error: latent binary loading missing after frailty validation".to_string()
            })?;
            let prepared = prepare_survival_time_stack(
                &age_entry,
                &age_exit,
                candidate,
                survival_mode,
                None,
                time_anchor,
                exact_derivative_guard,
                &time_build,
                None,
                Some(loading),
            )?;
            let time_p = prepared.time_design_exit.ncols();
            let time_initial_log_lambdas = prepared.time_initial_log_lambdas.clone();
            let time_block = TimeBlockInput {
                design_entry: prepared.time_design_entry.clone(),
                design_exit: prepared.time_design_exit.clone(),
                design_derivative_exit: prepared.time_design_derivative_exit.clone(),
                offset_entry: prepared.eta_offset_entry.clone(),
                offset_exit: prepared.eta_offset_exit.clone(),
                derivative_offset_exit: prepared.derivative_offset_exit.clone(),
                time_monotonicity:
                    crate::survival::location_scale::TimeBlockMonotonicity::EnforcedByCoordinateCone,
                penalties: prepared.time_penalties.clone(),
                nullspace_dims: prepared.time_nullspace_dims.clone(),
                initial_log_lambdas: time_initial_log_lambdas,
                initial_beta: Some(Array1::from_elem(time_p, 1e-4)),
            };
            Ok::<_, String>(LatentBinaryFitRequest {
                data: data.values.view(),
                spec: LatentBinaryTermSpec {
                    age_entry: age_entry.clone(),
                    age_exit: age_exit.clone(),
                    event_target: event.mapv(|v| if v >= 0.5 { 1 } else { 0 }),
                    weights: weights.clone(),
                    derivative_guard: exact_derivative_guard,
                    time_block,
                    unloaded_mass_entry: prepared.unloaded_mass_entry,
                    unloaded_mass_exit: prepared.unloaded_mass_exit,
                    meanspec: termspec.clone(),
                    mean_offset: threshold_offset.clone(),
                    baseline_config: candidate.clone(),
                },
                frailty: config.frailty.clone(),
                options: blockwise_fit_options(config),
            })
        };

    let baseline_cfg = if structural_only {
        // Structural formula validation must NOT fit. The baseline-θ resolution
        // for the location-scale and latent modes below is a real inner fit
        // (BFGS over `fit_model` evaluations); it only refines scale/shape, which
        // do not affect the request METADATA that validation reports
        // (family / model_class / schema / support). Running it here made
        // `validate_formula` — contractually "validate a formula against a
        // dataset WITHOUT fitting" — execute the full survival baseline workflow,
        // so a non-converging baseline fit surfaced as a *validation* error. Carry
        // the seed baseline config unchanged; the real fit path (this flag false)
        // still optimizes it below.
        baseline_cfg
    } else if matches!(
        survival_mode,
        SurvivalLikelihoodMode::Transformation
            | SurvivalLikelihoodMode::Weibull
            | SurvivalLikelihoodMode::MarginalSlope
    ) {
        baseline_cfg
    } else if baseline_cfg.target != SurvivalBaselineTarget::Linear
        && survival_mode == SurvivalLikelihoodMode::LocationScale
    {
        // Analytic θ-gradient path. The baseline configuration enters the
        // location-scale fit only through the three additive time-block
        // offsets (entry η, exit η, exit ∂η/∂t); at the converged β the
        // envelope theorem gives
        //
        //   d(NLL)/dθ_k = Σ_i r^(E)_i ∂o_E_i/∂θ_k
        //               + r^(X)_i ∂o_X_i/∂θ_k
        //               + r^(D)_i ∂o_D_i/∂θ_k
        //
        // where r^(*) are populated by
        // `SurvivalLocationScaleFamily::offset_channel_geometry` and the
        // partials by `baseline_offset_theta_partials`. When the inverse
        // link is probit/SAS/Mixture/etc., the location-scale family uses
        // the probit-channel baseline q(t) instead, so we contract against
        // `marginal_slope_baseline_offset_theta_partials` exactly as the
        // marginal-slope path does. BFGS w/ this analytic gradient
        // typically converges in ≲10 outer evaluations.
        let probit_channel =
            location_scale_uses_probit_survival_baseline(Some(&survival_inverse_link));
        // The search takes text, so a candidate's fit failure is kept typed here
        // (#2937). The outer engine never retries a thrown objective error: it
        // ends the search, so the kept failure is the one that stopped it.
        let candidate_failure = std::cell::RefCell::new(None::<FitFailure>);
        let stop_on = |failure: FitFailure| {
            let reason = failure.to_string();
            *candidate_failure.borrow_mut() = Some(failure);
            reason
        };
        let baseline_outcome = optimize_survival_baseline_config_with_gradient_only(
            &baseline_cfg,
            age_exit.view(),
            "workflow survival location-scale baseline",
            |candidate| {
                // A candidate spec that cannot be built is configuration.
                let request = build_location_scale_request(candidate).map_err(|reason| {
                    stop_on(FitFailure::from(WorkflowError::InvalidConfig { reason }))
                })?;
                let fit_result = fit_survival_location_scale_model(request).map_err(|failure| {
                    stop_on(failure.context("survival location-scale fit failed"))
                })?;
                // Warm-start the next probe's threshold / log-σ smoothing parameters
                // at the converged values for this probe.
                let threshold_rho = fit_result.fit.fit.lambdas_threshold().mapv(f64::ln);
                let log_sigma_rho = fit_result.fit.fit.lambdas_log_sigma().mapv(f64::ln);
                *location_scale_smoothing_warm_start.borrow_mut() =
                    Some((threshold_rho, log_sigma_rho));
                let residuals = &fit_result.fit.baseline_offset_residuals;
                let gradient = if probit_channel {
                    marginal_slope_baseline_chain_rule_gradient(
                        age_entry.view(),
                        age_exit.view(),
                        candidate,
                        residuals,
                    )
                    .map_err(|reason| stop_on(FitFailure::unclassified(reason)))?
                } else {
                    baseline_chain_rule_gradient(
                        age_entry.view(),
                        age_exit.view(),
                        // Location-scale has no interval channel; `residuals.right`
                        // is all-zero so `age_exit` is an unconsulted placeholder.
                        age_exit.view(),
                        candidate,
                        residuals,
                    )
                    .map_err(|reason| stop_on(FitFailure::unclassified(reason)))?
                }
                .ok_or_else(|| {
                    stop_on(FitFailure::invariant(
                        "workflow survival location-scale baseline unexpectedly has no theta gradient",
                    ))
                })?;
                // The envelope-theorem residual contraction is the exact
                // θ-gradient of the *profile penalized NLL* −ℓ + ½βᵀSβ at
                // converged (β̂, ρ̂). Optimizing `reml_score` (which includes
                // ½ log|S_λ| − ½ log|H| LAML corrections) against this
                // gradient would mismatch the cost surface, because the
                // log-determinant terms have their own θ-dependence through
                // H(β̂, θ). Use the matching profile-NLL cost here; the final
                // model refit downstream still picks ρ via the full REML
                // surface at the converged baseline θ. The gradient belongs to
                // the mode, so the cost reads the mode's log-likelihood, not the
                // one at a published posterior mean (gam#2921).
                let log_likelihood_at_mode = fit_result.fit.fit.log_likelihood_at_mode();
                let profile_cost =
                    -log_likelihood_at_mode + 0.5 * fit_result.fit.fit.stable_penalty_term;
                if !profile_cost.is_finite() {
                    return Err(stop_on(FitFailure::numerical(format!(
                        "workflow survival location-scale baseline: non-finite profile cost \
                         (log_likelihood_at_mode={}, stable_penalty_term={}, cost={})",
                        log_likelihood_at_mode,
                        fit_result.fit.fit.stable_penalty_term,
                        profile_cost
                    ))));
                }
                Ok((profile_cost, gradient))
            },
        );
        match baseline_outcome {
            Ok(baseline) => baseline,
            Err(search) => {
                return Err(match candidate_failure.take() {
                    // A candidate's fit stopped the search: raise that failure
                    // under its category.
                    Some(failure) => WorkflowError::from(failure),
                    // Otherwise the search's own typed verdict, or its
                    // configuration refusal, stands.
                    None => search,
                });
            }
        }
    } else {
        // A latent survival or binary fit selects its baseline chart together with
        // ρ on the one LAML criterion (#2714).
        baseline_cfg
    };

    let request = match survival_mode {
        SurvivalLikelihoodMode::Transformation | SurvivalLikelihoodMode::Weibull => {
            if config.noise_offset_column.is_some() {
                return Err(WorkflowError::InvalidConfig {
                    reason:
                        "noise_offset_column is supported only for survival location-scale or marginal-slope"
                            .to_string(),
                }
                .into());
            }
            let weibull_seed = if survival_mode == SurvivalLikelihoodMode::Weibull
                && effective_timewiggle.is_none()
            {
                let scale = config
                    .baseline_scale
                    .unwrap_or_else(|| positive_survival_time_seed(&age_exit));
                let shape = config.baseline_shape.unwrap_or(1.0);
                if !scale.is_finite() || scale <= 0.0 || !shape.is_finite() || shape <= 0.0 {
                    return Err(WorkflowError::InvalidConfig {
                        reason:
                            "weibull survival fit requires finite positive baseline_scale and baseline_shape"
                                .to_string(),
                    }
                    .into());
                }
                Some((scale, shape))
            } else {
                None
            };
            FitRequest::SurvivalTransformation(SurvivalTransformationFitRequest {
                data: data.values.view(),
                spec: SurvivalTransformationTermSpec {
                    age_entry: age_entry.clone(),
                    age_exit: age_exit.clone(),
                    event_target: event_codes.clone(),
                    weights: weights.clone(),
                    covariate_spec: termspec.clone(),
                    covariate_offset: threshold_offset.clone(),
                    baseline_cfg,
                    likelihood_mode: survival_mode,
                    time_anchor,
                    time_build: time_build.clone(),
                    timewiggle: effective_timewiggle.clone(),
                    weibull_seed,
                    penalty_block_gamma_priors: config.penalty_block_gamma_priors.clone(),
                },
                persistent_warm_start_store: config.persistent_warm_start_store.clone(),
            })
        }
        SurvivalLikelihoodMode::LocationScale => {
            FitRequest::SurvivalLocationScale(build_location_scale_request(&baseline_cfg)?)
        }
        SurvivalLikelihoodMode::MarginalSlope => {
            FitRequest::SurvivalMarginalSlope(build_marginal_slope_request()?)
        }
        SurvivalLikelihoodMode::Latent => {
            FitRequest::LatentSurvival(build_latent_survival_request(&baseline_cfg)?)
        }
        SurvivalLikelihoodMode::LatentBinary => {
            FitRequest::LatentBinary(build_latent_binary_request(&baseline_cfg)?)
        }
    };

    Ok(MaterializedModel {
        request,
        inference_notes,
        unidentified_scalar_terms,
        survival_time_basis: Some(
            crate::survival::construction::SavedSurvivalTimeBasis::from_build(
                &time_build,
                time_anchor,
            ),
        ),
    })
}
