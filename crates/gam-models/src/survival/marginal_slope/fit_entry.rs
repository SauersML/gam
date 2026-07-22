//! The public fitting entry point `fit_survival_marginal_slope_terms`.

use super::*;

pub fn fit_survival_marginal_slope_terms(
    data: ArrayView2<'_, f64>,
    spec: SurvivalMarginalSlopeTermSpec,
    options: &BlockwiseFitOptions,
    kappa_options: &SpatialLengthScaleOptimizationOptions,
) -> Result<SurvivalMarginalSlopeFitResult, String> {
    // The outer search is bounded by deterministic work (iteration/cycle caps
    // and the seed-screening cascade budget), not by wall-clock time (#2055):
    // clipping a fit by elapsed time is non-deterministic and machine-dependent,
    // so a slow-to-converge fit is fixed or bounded by work, never by a timer.
    fit_survival_marginal_slope_terms_impl(data, spec, options, kappa_options)
}

pub(crate) fn fit_survival_marginal_slope_terms_impl(
    data: ArrayView2<'_, f64>,
    spec: SurvivalMarginalSlopeTermSpec,
    options: &BlockwiseFitOptions,
    kappa_options: &SpatialLengthScaleOptimizationOptions,
) -> Result<SurvivalMarginalSlopeFitResult, String> {
    let fit_started = std::time::Instant::now();
    let mut spec = spec;
    validate_spec(&spec)?;
    if spec.base_link != InverseLink::Standard(StandardLink::Probit) {
        return Err(SurvivalMarginalSlopeError::UnsupportedConfiguration {
            reason: format!(
                "survival-marginal-slope currently supports only probit base_link, got {:?}",
                spec.base_link
            ),
        }
        .into());
    }
    install_time_nullspace_shrinkage_penalty(&mut spec.time_block)?;
    let (z_standardized, z_normalization) = standardize_latent_z_matrix_with_policy(
        &spec.z,
        &spec.weights,
        "survival-marginal-slope",
        &spec.latent_z_policy,
    )?;
    spec.z = z_standardized;
    let score_covariance = marginal_slope_covariance_from_scores(spec.z.view(), &spec.weights)?;
    let z_primary = spec.z.column(0).to_owned();
    let n = spec.age_entry.len();
    let (initial_sigma, learned_sigma_initial, learned_log_sigma_coordinate) = match &spec.frailty {
        FrailtySpec::GaussianShift {
            scale: FrailtyScale::Fixed { sigma },
        } => (Some(*sigma), None, None),
        FrailtySpec::GaussianShift {
            scale: scale @ FrailtyScale::Learned { initial_sigma },
        } => (
            Some(*initial_sigma),
            Some(*initial_sigma),
            scale.learned_log_sigma_coordinate(),
        ),
        FrailtySpec::None => (None, None, None),
        FrailtySpec::HazardMultiplier { .. } => {
            return Err(SurvivalMarginalSlopeError::InvalidInput {
                reason:
                    "internal: validate_spec should have rejected unsupported marginal-slope frailty"
                        .to_string(),
            }
            .into());
        }
    };
    let (baseline_initial_theta, baseline_lower_theta, baseline_upper_theta) =
        match &spec.baseline_hyper {
            SurvivalMarginalSlopeBaselineHyperSpec::Linear { .. } => {
                (Vec::new(), Vec::new(), Vec::new())
            }
            SurvivalMarginalSlopeBaselineHyperSpec::Nonlinear { chart } => {
                let (lower, upper) = chart.theta_bounds();
                (chart.initial_theta().to_vec(), lower.to_vec(), upper.to_vec())
            }
        };
    let probit_scale = probit_frailty_scale(initial_sigma);
    let baseline_started = std::time::Instant::now();
    let baseline_slope = pooled_survival_baseline(
        &spec.event_target,
        &spec.weights,
        &z_primary,
        &spec.time_block.offset_entry,
        &spec.time_block.offset_exit,
        &spec.time_block.derivative_offset_exit,
        probit_scale,
    );
    log::info!(
        "[survival-marginal-slope] baseline seed slope={:.6e} elapsed={:.3}s",
        baseline_slope,
        baseline_started.elapsed().as_secs_f64(),
    );

    let logslope_specs_input = spec
        .logslopespecs
        .clone()
        .unwrap_or_else(|| vec![spec.logslopespec.clone()]);
    if logslope_specs_input.len() != spec.z.ncols() && logslope_specs_input.len() != 1 {
        return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
            reason: format!(
                "survival marginal-slope expected either one shared logslope spec or one spec per z coordinate (K={}); got {}",
                spec.z.ncols(),
                logslope_specs_input.len()
            ),
        }
        .into());
    }
    let mut design_specs = Vec::with_capacity(1 + logslope_specs_input.len());
    design_specs.push(spec.marginalspec.clone());
    design_specs.extend(logslope_specs_input.iter().cloned());
    // gam#979: give the marginal + logslope smooth surfaces the Marra & Wood
    // double penalty so their polynomial-trend null space is identified rather
    // than left flat. Without it that trend direction is a large-signal /
    // tiny-curvature mode that deadlocks the inner joint-Newton (the spectral
    // step drops it #1082 while the certificate requires it #1449) — the
    // survival marginal-slope hang. Applied before the build so the flag is
    // frozen into `joint_specs` and honoured by every subsequent probe / frozen
    // / kappa rebuild. Mirrors the time block's
    // `install_time_nullspace_shrinkage_penalty`, via the ordinary builder so
    // the layered penalty representation stays self-consistent.
    for surface_spec in design_specs.iter_mut() {
        enable_surface_identifiability_double_penalty(surface_spec);
    }
    let (_raw_joint_designs, mut joint_specs) =
        build_term_collection_designs_and_freeze_joint(data, &design_specs)
            .map_err(|e| e.to_string())?;
    // Rebuild the probe designs from the frozen `joint_specs` so the probe's
    // penalty topology matches the topology produced by every other build path
    // in this optimization. The spatial optimizer's own bootstrap inside
    // `optimize_spatial_length_scale_exact_joint` and every subsequent
    // kappa-driven rebuild feed the basis builder the captured
    // `FrozenTransform` identifiability. Applying that captured transform
    // changes the coefficient chart in which every penalty is represented.
    // Without this rebuild, the probe's
    // penalty count overshoots every subsequent evaluator's measurement of
    // the frozen build, and `evaluate_custom_family_joint_hyper` refuses with
    // a "joint hyper rho dimension mismatch". Mirrors the CTN- and BMS-side
    // fixes in `fit_transformation_normal` and `fit_bernoulli_marginal_slope_terms`.
    let (mut joint_designs, _) = build_term_collection_designs_and_freeze_joint(data, &joint_specs)
        .map_err(|e| format!("failed to rebuild frozen probe SMGS joint designs: {e}"))?;
    let marginal_design = joint_designs.remove(0);
    let marginalspec_boot = joint_specs.remove(0);
    let (logslope_design, logslopespec_boot, logslope_topology) =
        combine_logslope_surface_designs(joint_designs, &joint_specs)?;
    spec.marginal_offset = marginal_design
        .compose_offset(
            spec.marginal_offset.view(),
            "survival marginal-slope marginal block",
        )
        .map_err(|error| error.to_string())?;
    spec.logslope_offset = logslope_design
        .compose_offset(
            spec.logslope_offset.view(),
            "survival marginal-slope logslope block",
        )
        .map_err(|error| error.to_string())?;
    let common_logslope_offset = &spec.logslope_offset + baseline_slope;
    if logslope_topology.is_per_score() && logslope_topology.score_count() != spec.z.ncols() {
        return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
            reason: format!(
                "survival marginal-slope has {} per-score logslope channels but latent score dimension K={}",
                logslope_topology.score_count(),
                spec.z.ncols(),
            ),
        }
        .into());
    }
    let time_penalties_len = spec.time_block.penalties.len();
    let mut cross_block_warnings: Vec<CrossBlockIdentifiabilityWarning> = Vec::new();
    // Cross-block W metric: build the survival rigid pooled-probit pilot η
    // ONCE and use the survival row neg-log Hessian diagonal wrt η₁ at the
    // pilot for both flex anchor orthogonalisations below
    // (`survival_pilot_irls_row_metric_at_eta` — see its doc for the exact
    // formula; matches `u2_eta1` in `row_primary_closed_form`). Using
    // `&spec.weights` (uniform sample weights) instead would make A and C̃
    // merely Euclidean-orthogonal — at PIRLS time `Aᵀ W_pirls C̃ ≠ 0`, the
    // joint Hessian carries a near-null direction along the W-metric alias,
    // and REML can drive the flex block's λ small enough that the alias
    // direction's joint Hessian eigenvalue collapses (the `rho≈2.0`,
    // constant `step_inf`, growing `beta_inf` runaway documented for BMS).
    // The earlier copy of BMS's `w · φ(η)² / (Φ(η)(1−Φ(η)))` was a probit-
    // shaped proxy, not the survival row curvature; using the actual
    // survival curvature here matches the inner product PIRLS sees on this
    // family. Analog of bernoulli_marginal_slope.rs:19213-19214 + :19355-19356.
    // Build the location anchor design once, ahead of both the cross-block
    // W-metric pilot and the cross-block residualisation calls below. The
    // pilot needs it to compute a one-step IRLS refinement of η₁ that
    // varies per row; reusing it for the residualisation calls keeps the
    // anchor matmul width consistent between fit time and predict time
    // (`predict.rs` ~1158 enforces `[time_exit | marginal]` width on the
    // saved `anchor_correction_matrix`).
    let location_anchor_design = DesignMatrix::hstack(vec![
        spec.time_block.design_exit.clone(),
        marginal_design.design.clone(),
    ])
    .map_err(|e| {
        format!(
            "survival marginal-slope cross-block anchor stack failed to concatenate time + marginal design at training rows: {e}"
        )
    })?;
    // Non-rigid pilot η₁ via one IRLS step on the rigid joint design
    // `[T_exit | M | G]`. The offset-only `survival_rigid_pilot_eta` was a
    // near-constant scalar when training offsets are uniform (the typical
    // large-scale case), which collapses the cross-block W metric onto a
    // single direction and lets `link_dev` alias `score_warp_dev` along
    // the scalar PRS-axis path documented in the original audit (item #3
    // of the principled fix: link_dev seed must reflect actual fitted
    // exit-location and logslope surfaces, not just data offsets). One
    // Newton step is sufficient for the cross-block residualisation: we
    // need a per-row-varying η₁ that respects event/weight structure, not
    // a converged β.
    let (cross_block_pilot_eta, pilot_logslope_beta) = survival_nonrigid_pilot_eta(
        n,
        &location_anchor_design,
        &logslope_design.design,
        &z_primary,
        &spec.time_block.offset_exit,
        &spec.marginal_offset,
        &spec.logslope_offset,
        baseline_slope,
        &spec.weights,
        &spec.event_target,
        probit_scale,
    )?;
    let cross_block_pilot_w = survival_pilot_irls_row_metric_at_eta(
        &cross_block_pilot_eta,
        &spec.weights,
        &spec.event_target,
    )
    .map_err(|e| format!("survival cross-block W metric construction: {e}"))?;
    // Absorbed Stage-1 influence columns `Z̃_infl` (#461, design §3). When the
    // workflow chained a CTN Stage-1 into this marginal-slope fit,
    // `spec.score_influence_jacobian` carries the out-of-fold `J = ∂z/∂θ₁`. The
    // realized leakage directions `Z_infl = diag(s_f·β̂₀)·J` are residualized
    // against the marginal location span in the rigid-pilot row metric
    // (`cross_block_pilot_w`) — keeping the logslope-aligned component — and
    // hosted as a dedicated additive absorber block whose coefficient `γ` shifts
    // the de-nested observed index `η₁` by `+Z̃_infl·γ`. β̂₀(x_i) is the
    // rigid-pilot logslope `baseline_slope + logslope_offset[i]`; `s_f =
    // probit_scale`. The math (residualize-vs-marginal/retain-logslope +
    // fixed-ridge absorber) is the single source of truth shared with the BMS
    // family via `marginal_slope_orthogonal`; survival differs only in the host
    // structure — a dedicated `η₁` channel rather than BMS's widened marginal
    // index, because the survival marginal block feeds the time-quantile
    // location `q·c(g)` (scaled), not a flat additive index. `None` ⇒ raw `z`,
    // and the free `score_warp` spline below is the x-free-column fallback.
    let influence_absorber_residualized: Option<Array2<f64>> = if let Some(jac) = spec
        .score_influence_jacobian
        .as_ref()
        .filter(|jac| jac.ncols() > 0)
    {
        // A zero-column Jacobian carries no leakage directions ⇒ no absorber.
        use crate::marginal_slope_orthogonal::residualized_influence_block;
        let marginal_dense = marginal_design
            .design
            .try_to_dense_by_chunks("survival marginal-slope influence-absorber marginal span")?;
        // `β̂₀(x_i)` is the rigid-pilot logslope; `s_f = probit_scale`; `z_primary`
        // is the OOF latent z on these rows.
        let rigid_logslope_at_rows = &spec.logslope_offset + baseline_slope;
        // Z̃_infl = residualize(diag(s_f·β̂₀)·J, marginal, W) — the combined core
        // builder (single source of truth shared with the BMS absorber site). It
        // takes the raw n×p₁ J + OOF z and encapsulates the full §3 sequence: build
        // Z_infl, derive the weighted marginal-Gram ridge internally (max diag·1e-10,
        // floored 1e-12), residualize, and finite-check (Err on non-finite), so this
        // caller passes no ε and propagates the error.
        let residualized = residualized_influence_block(
            jac,
            &z_primary,
            &rigid_logslope_at_rows,
            probit_scale,
            marginal_dense.view(),
            &cross_block_pilot_w,
        )
        .map_err(|reason| SurvivalMarginalSlopeError::NumericalFailure { reason })?;
        Some(residualized)
    } else {
        None
    };
    // `location_anchor_design` was built above (alongside the non-rigid
    // pilot η) and is reused here for the cross-block residualisation
    // calls. Keeping the construction at one site means the
    // predict-time `anchor_correction_matrix` width contract — survival
    // prediction routes through `BernoulliMarginalSlopePredictor`
    // (`survival_predict.rs:2458`) which passes
    // `[time_block.design_exit | timewiggle_exit | cov_design]` and
    // `predict.rs:1158` treats AS-IS as the parametric anchor — is
    // satisfied by a single source of truth.
    // Score-warp: build the scalar base DeviationPrepared, apply cross-
    // block identifiability reparameterisation against the parametric
    // anchor union (marginal + logslope) on the underlying runtime, then
    // re-stripe across z coordinates. Reparameterising on the scalar base
    // BEFORE the direct-sum striping is the principled order: the per-z
    // stripes share a single runtime, so installing the W-metric residual
    // (`anchor_residual` + cached `anchor_rows_at_training`) once means
    // every stripe inherits the same orthogonal-complement basis when
    // `design_at_training_with_residual` is called. Without this, the
    // score-warp block carries its own constant / low-order η-polynomial
    // direction on every z stripe, producing the alias pencil
    //   score_warp_dev[k] ≡ marginal_surface[m] ≡ logslope_surface[ℓ]
    // and (against the already-reparameterised link-dev) the
    //   score_warp_dev[k] ≡ link_dev[k]
    // overlap chain documented by the identifiability audit.
    let score_warp_prepared = if let Some(cfg) = spec.score_warp.as_ref() {
        let score_dim = spec.z.ncols();
        if score_dim == 0 {
            return Err(SurvivalMarginalSlopeError::InvalidInput {
                reason: "survival score-warp requires at least one z coordinate".to_string(),
            }
            .into());
        }
        let mut base = build_score_warp_deviation_block_from_seed(&z_primary, cfg)?;
        let parametric_anchors: [(&DesignMatrix, ParametricAnchorBlock); 2] = [
            (&location_anchor_design, ParametricAnchorBlock::Marginal),
            (&logslope_design.design, ParametricAnchorBlock::Logslope),
        ];
        let outcome = install_compiled_flex_block_into_runtime(
            &mut base,
            &z_primary,
            cfg,
            &parametric_anchors,
            &[],
            &cross_block_pilot_w,
        )?;
        match outcome {
            FlexCompileOutcome::Reparameterised => Some(stripe_score_warp_across_z_coords(
                base.block,
                base.runtime,
                &spec.z,
            )?),
            FlexCompileOutcome::FullyAliased { reason } => {
                // Record via the structured channel. The block is still
                // included with its original (non-compiled) design so the
                // unified audit's canonicalize_for_identifiability sees it
                // and attributes the drop to score_warp_dev via
                // dropped_columns (gauge_priority=80 is below marginal=150
                // and logslope=120 so RRQR correctly demotes score_warp_dev).
                // No family-side log.warn — the audit's DroppedColumn record
                // IS the authoritative structured report.
                cross_block_warnings.push(CrossBlockIdentifiabilityWarning {
                    candidate_label: "score_warp",
                    anchor_summary: "marginal+logslope".to_string(),
                    reason,
                });
                Some(stripe_score_warp_across_z_coords(
                    base.block,
                    base.runtime,
                    &spec.z,
                )?)
            }
        }
    } else {
        None
    };
    let link_dev_prepared = if let Some(cfg) = spec.link_dev.as_ref() {
        // q0_seed and the cross-block W-metric pilot η are intentionally the
        // same vector: the link-deviation basis is anchored at this η, and
        // the orthogonalisation metric uses the IRLS Hessian row weight at
        // the same η so `Aᵀ W C̃ = 0` holds in the inner product the joint
        // Hessian sees during PIRLS.
        let q0_seed = cross_block_pilot_eta.clone();
        let padded_seed = padded_deviation_seed(&q0_seed, 1.0, 0.5);
        let mut prepared = build_link_deviation_block_from_knots_design_seed_and_weights(
            &padded_seed,
            &q0_seed,
            cfg,
        )?;
        // Cross-block identifiability: residualise the link-deviation
        // basis against the parametric anchor union (marginal + logslope
        // designs) at training rows so its column span is orthogonal
        // to span(X_marginal, X_logslope). Without this, the link-dev
        // basis's constant / low-order η-polynomial directions are
        // exactly the constant columns carried by marginal_surface and
        // logslope_surface, producing the alias pencil
        //   link_dev[k] ≡ marginal_surface[m] ≡ logslope_surface[ℓ]
        // documented by the identifiability audit (joint rank collapse
        // from 51 → 38 on the large-scale binary-outcome fit). After
        // the W-metric eigendecomposition installs `T_lw` on the
        // DeviationRuntime, the joint design has full numerical column
        // rank with respect to (marginal, logslope) and the joint
        // penalised Hessian satisfies `σ_min(joint H + S) ≥ λ_min(S) > 0`
        // for every β. This mirrors the BMS construction site at
        // bernoulli_marginal_slope.rs:18236, transplanted here because
        // SMGS had no cross-block reparam call. The W metric routed in
        // here is the survival row neg-log Hessian diagonal wrt η₁ at the
        // rigid pooled-probit pilot η (see `cross_block_pilot_w` above —
        // matches `u2_eta1` in `row_primary_closed_form`); this is the
        // inner product the joint penalised Hessian sees during PIRLS, so
        // `Aᵀ W C̃ = 0` after the reparam survives into PIRLS rather than
        // holding only under uniform `spec.weights`.
        // Thread the now-reparameterised score-warp basis at training rows
        // as a flex-evaluation anchor so the link-deviation basis is jointly
        // orthogonal to span(marginal, logslope, score_warp). Mirrors BMS at
        // bernoulli_marginal_slope.rs:18291-18307. For the per-z striped
        // score-warp, only the primary-coordinate basis is needed as a flex
        // anchor (score-warp's per-z stripes share a single underlying
        // basis, all in the same orthogonal complement of the parametric
        // anchors after the score-warp reparam above), so we evaluate the
        // reparameterised runtime at z_primary.
        let score_warp_anchor_design = score_warp_prepared
            .as_ref()
            .map(|sw| sw.runtime.design_at_training_with_residual(&z_primary))
            .transpose()?;
        let parametric_anchors: [(&DesignMatrix, ParametricAnchorBlock); 2] = [
            (&location_anchor_design, ParametricAnchorBlock::Marginal),
            (&logslope_design.design, ParametricAnchorBlock::Logslope),
        ];
        let flex_anchor_slot: Option<&Array2<f64>> = score_warp_anchor_design.as_ref();
        let flex_anchors: Vec<&Array2<f64>> = flex_anchor_slot.into_iter().collect();
        let outcome = install_compiled_flex_block_into_runtime(
            &mut prepared,
            &q0_seed,
            cfg,
            &parametric_anchors,
            &flex_anchors,
            &cross_block_pilot_w,
        )?;
        match outcome {
            FlexCompileOutcome::Reparameterised => Some(prepared),
            FlexCompileOutcome::FullyAliased { reason } => {
                // Record via the structured channel. Keep the original
                // (non-compiled) design so the unified audit sees link_dev
                // with its original columns and attributes the alias drop
                // via dropped_columns (gauge_priority=60 < marginal=150 /
                // logslope=120 so RRQR correctly demotes link_dev).
                cross_block_warnings.push(CrossBlockIdentifiabilityWarning {
                    candidate_label: "link_deviation",
                    anchor_summary: "marginal+logslope".to_string(),
                    reason,
                });
                Some(prepared)
            }
        }
    } else {
        None
    };
    // Penalty seeds for the flex/aux blocks beyond the core (time/marginal/
    // logslope). The absorbed influence block (#461) contributes ONE trailing
    // REML-learned identity penalty on γ: the outer optimizer selects the
    // absorber precision like any other random-effect variance, seeded at the
    // ln(n) leakage scale (SPEC: shrinkage is explicit or REML-selected, never
    // a pinned magic constant). The absorber columns are residualized against
    // the marginal span, so a small learned λ cannot absorb genuine β(x)
    // signal; a large learned λ recovers the null correction.
    let extra_rho0 = {
        let mut out = Vec::new();
        if let Some(ref prepared) = score_warp_prepared {
            out.extend(std::iter::repeat_n(0.0, prepared.block.penalties.len()));
        }
        if let Some(ref prepared) = link_dev_prepared {
            out.extend(std::iter::repeat_n(0.0, prepared.block.penalties.len()));
        }
        if influence_absorber_residualized.is_some() {
            // The absorber's single learned ridge sits at the trailing extra
            // slot; the seed is clamped into the outer ρ box.
            out.push(
                crate::marginal_slope_orthogonal::influence_absorber_log_lambda(n)
                    .clamp(-12.0, 12.0),
            );
        }
        out
    };
    let core_rho0_seed: Vec<f64> = {
        let mut seeds = Vec::with_capacity(
            time_penalties_len + marginal_design.penalties.len() + logslope_design.penalties.len(),
        );
        seeds.extend(block_log_lambda_seeds(
            &spec.time_block.design_exit,
            spec.time_block.penalties.iter(),
        ));
        seeds.extend(block_log_lambda_seeds(
            &marginal_design.design,
            marginal_design.penalties.iter().map(|bp| &bp.local),
        ));
        seeds.extend(block_log_lambda_seeds(
            &logslope_design.design,
            logslope_design.penalties.iter().map(|bp| &bp.local),
        ));
        seeds
    };
    let tiny_fixed_kappa_options = if n < 1_000 && kappa_options.enabled {
        let mut opts = kappa_options.clone();
        opts.enabled = false;
        log::info!(
            "[survival-marginal-slope/kappa] fixed-bootstrap-kappa tiny-fit policy n={} threshold=1000",
            n,
        );
        Some(opts)
    } else {
        None
    };
    let kappa_options_effective = tiny_fixed_kappa_options.as_ref().unwrap_or(kappa_options);
    let setup = joint_setup(
        data,
        time_penalties_len,
        &marginalspec_boot,
        marginal_design.penalties.len(),
        &logslopespec_boot,
        logslope_design.penalties.len(),
        &core_rho0_seed,
        &extra_rho0,
        &baseline_initial_theta,
        &baseline_lower_theta,
        &baseline_upper_theta,
        learned_log_sigma_coordinate,
        kappa_options_effective,
    )
    .map_err(|error| error.to_string())?;

    let hints = RefCell::new(ThetaHints::default());
    // #808 operating-point warm start for the logslope block. The inner
    // joint-Newton seeds each block at `spec.initial_beta` (→ `hints.logslope_beta`
    // via `build_logslope_blockspec`). At the default `g = 0` seed the logslope
    // block is W-null (the slope-channel IRLS weight vanishes at the null slope),
    // so the inner cannot take its first step and freezes (the #808 stall). Seed
    // it instead at the one-step non-rigid pilot's logslope coefficients, which
    // put `g` at the operating point (`g ≈ 0.3`) where the slope channel carries
    // information and the block is full-rank — breaking the chicken-and-egg so the
    // inner moves and converges to the true data optimum. It is only a warm start,
    // so the converged β is the data optimum (zero bias; the log-slope estimand is
    // recovered, NOT dropped or pinned to zero). Width-guarded against any
    // logslope design rebuild.
    if pilot_logslope_beta.len() == logslope_design.design.ncols()
        && pilot_logslope_beta.iter().all(|v| v.is_finite())
    {
        hints.borrow_mut().logslope_beta = Some(pilot_logslope_beta.clone());
    }
    let exact_warm_start = RefCell::new(None::<CustomFamilyWarmStart>);
    // Outer ρ-cache β-seed staging slot. The spatial-joint optimizer fires
    // `seed_inner_beta_fn` on a cache hit before any eval has run at the
    // restored ρ. Per-block widths are only known once `build_blocks(rho,…)`
    // runs, so we stash the flat β here and the eval closures promote it
    // into `exact_warm_start` on the first invocation.
    let pending_beta_seed = RefCell::new(None::<Array1<f64>>);
    // Monotonic per-outer-eval counter used to populate
    // `BlockwiseFitOptions::outer_eval_context` so downstream
    // auto-subsample install paths key on (rho, eval_id) instead of
    // the inner β. Distinct outer derivative evaluations always get a
    // distinct eval_id; the contained `EvalScope` distinguishes the
    // outer derivative call from inner trial line-search calls (which
    // copy this id but flip the scope to `InnerCoefficient`).
    let outer_eval_counter = std::cell::Cell::new(0usize);

    let event = Arc::new(spec.event_target.clone());
    let weights = Arc::new(spec.weights.clone());
    let z = Arc::new(spec.z.clone());
    let derivative_guard = spec.derivative_guard;

    let design_entry = spec.time_block.design_entry.clone();
    let design_exit = spec.time_block.design_exit.clone();
    let design_derivative_exit = spec.time_block.design_derivative_exit.clone();
    let offset_entry = Arc::new(spec.time_block.offset_entry.clone());
    let offset_exit = Arc::new(spec.time_block.offset_exit.clone());
    let derivative_offset_exit = Arc::new(spec.time_block.derivative_offset_exit.clone());
    let time_block_ref = spec.time_block.clone();
    let score_warp_runtime = score_warp_prepared.as_ref().map(|p| p.runtime.clone());
    let link_dev_runtime = link_dev_prepared.as_ref().map(|p| p.runtime.clone());
    let derived_time_wiggle_ncols = spec
        .timewiggle_block
        .as_ref()
        .map(|timewiggle| time_wiggle_basis_ncols(&timewiggle.knots, timewiggle.degree))
        .transpose()?;
    // Coordinate-cone time bases already encode monotonicity as β >= 0:
    // validation proved D >= 0 and offsets absorb the derivative guard. Emitting
    // row-wise `D β + o >= guard` constraints here duplicates the same condition
    // as hundreds of dense rows and forces the generic active-set QP path. Use
    // a single identity cone instead so the custom-family solver recognizes the
    // simple lower-bound problem.
    let time_linear_constraints = match spec.time_block.time_monotonicity {
        monotonicity if monotonicity.is_coordinate_cone() => {
            let p_total = design_exit.ncols();
            LinearInequalityConstraints::from_per_coordinate_lower_bounds(&Array1::<f64>::zeros(
                p_total,
            ))
        }
        _ => {
            let derivative_guard_constraints = time_derivative_guard_constraints(
                &design_derivative_exit,
                derivative_offset_exit.as_ref(),
                derivative_guard,
            )?;
            append_timewiggle_tail_nonnegative_constraints(
                derivative_guard_constraints,
                design_exit.ncols(),
                derived_time_wiggle_ncols.unwrap_or(0),
            )?
        }
    };

    let intercept_warm_starts = new_intercept_warm_start_cache(n);
    let initial_hyper_theta = setup.theta0();
    let family_coordinate_start = setup.rho_dim() + setup.log_kappa_dim();
    let baseline_axis_count = baseline_initial_theta.len();
    let sigma_coordinate = learned_sigma_initial
        .is_some()
        .then_some(family_coordinate_start + baseline_axis_count);
    let sigma_from_theta = |theta: &Array1<f64>| -> Result<Option<f64>, String> {
        match sigma_coordinate {
            Some(axis) => theta
                .get(axis)
                .copied()
                .ok_or_else(|| {
                    format!(
                        "survival marginal-slope theta has {} coordinates, missing learned log-sigma axis {axis}",
                        theta.len()
                    )
                })
                .map(|log_sigma| Some(log_sigma.exp())),
            None => Ok(initial_sigma),
        }
    };
    let family_hyper_from_theta =
        |theta: &Array1<f64>| -> Result<SurvivalMarginalSlopeFamilyHyperState, String> {
            let baseline_end = family_coordinate_start + baseline_axis_count;
            if theta.len() < baseline_end {
                return Err(format!(
                    "survival marginal-slope theta has {} coordinates, expected at least {baseline_end} to realize the baseline chart",
                    theta.len()
                ));
            }
            let baseline_geometry = match &spec.baseline_hyper {
                SurvivalMarginalSlopeBaselineHyperSpec::Linear { .. } => None,
                SurvivalMarginalSlopeBaselineHyperSpec::Nonlinear { chart } => {
                    let baseline_theta = theta
                        .slice(s![family_coordinate_start..baseline_end])
                        .to_owned();
                    Some(Arc::new(chart.evaluate(&baseline_theta)?))
                }
            };
            let learned_log_sigma = sigma_coordinate
                .map(|axis| {
                    theta.get(axis).copied().ok_or_else(|| {
                        format!(
                            "survival marginal-slope theta has {} coordinates, missing learned log-sigma axis {axis}",
                            theta.len()
                        )
                    })
                })
                .transpose()?;
            SurvivalMarginalSlopeFamilyHyperState::new(
                baseline_geometry,
                learned_log_sigma,
            )
        };
    // FlexActivation::OffForRigidPilot forces the rigid warm-start to construct
    // a family with no score_warp / link_dev runtimes and no flex blocks. That
    // is the only way to guarantee the pilot does not enter the survival flex
    // exact-Joint-Newton path. A boolean here would be too easy to flip
    // accidentally; the named enum makes the intent and audit obvious at every
    // call site.
    let make_family = |marginal_design: &TermCollectionDesign,
                       logslope_design: &TermCollectionDesign,
                       theta: &Array1<f64>,
                       flex: FlexActivation|
     -> Result<SurvivalMarginalSlopeFamily, String> {
        let family_hyper = family_hyper_from_theta(theta)?;
        let sigma = sigma_from_theta(theta)?;
        let (family_offset_entry, family_offset_exit, family_derivative_offset_exit) =
            match family_hyper.baseline_geometry.as_ref() {
                Some(geometry) => (
                    Arc::new(geometry.offset_entry.clone()),
                    Arc::new(geometry.offset_exit.clone()),
                    Arc::new(geometry.derivative_offset_exit.clone()),
                ),
                None => (
                    Arc::clone(&offset_entry),
                    Arc::clone(&offset_exit),
                    Arc::clone(&derivative_offset_exit),
                ),
            };
        let (score_warp_active, link_dev_active) = match flex {
            FlexActivation::OffForRigidPilot => (None, None),
            FlexActivation::On => (score_warp_runtime.clone(), link_dev_runtime.clone()),
        };
        // The absorber is suppressed during the rigid-pilot pass: its pilot
        // logslope β̂₀ and the residualization W metric are *derived from* that
        // pilot, so it can only enter the full (non-rigid) fit (mirror of the
        // score_warp/link_dev `FlexActivation` gating above).
        let influence_absorber_active = match flex {
            FlexActivation::OffForRigidPilot => None,
            FlexActivation::On => influence_absorber_residualized.clone(),
        };
        let logslope_layout = logslope_topology
            .materialize_identity(logslope_design.design.clone(), &common_logslope_offset)?;
        logslope_layout.validate_for(spec.z.ncols())?;
        Ok(SurvivalMarginalSlopeFamily {
            n,
            event: Arc::clone(&event),
            weights: Arc::clone(&weights),
            z: Arc::clone(&z),
            score_covariance: score_covariance.clone(),
            gaussian_frailty_sd: sigma,
            family_hyper,
            derivative_guard,
            design_entry: design_entry.clone(),
            design_exit: design_exit.clone(),
            design_derivative_exit: design_derivative_exit.clone(),
            offset_entry: family_offset_entry,
            offset_exit: family_offset_exit,
            derivative_offset_exit: family_derivative_offset_exit,
            marginal_design: marginal_design.design.clone(),
            logslope_layout,
            score_warp: score_warp_active,
            link_dev: link_dev_active,
            influence_absorber: influence_absorber_active,
            time_linear_constraints: time_linear_constraints.clone(),
            time_wiggle_knots: spec.timewiggle_block.as_ref().map(|w| w.knots.clone()),
            time_wiggle_degree: spec.timewiggle_block.as_ref().map(|w| w.degree),
            time_wiggle_ncols: derived_time_wiggle_ncols.unwrap_or(0),
            intercept_warm_starts: Some(Arc::clone(&intercept_warm_starts)),
            auto_subsample_phase_counter: Arc::new(AtomicUsize::new(0)),
            auto_subsample_last_rho: Arc::new(Mutex::new(None)),
        })
    };

    let build_blocks = |rho: &Array1<f64>,
                        marginal_design: &TermCollectionDesign,
                        logslope_design: &TermCollectionDesign,
                        flex: FlexActivation|
     -> Result<Vec<ParameterBlockSpec>, String> {
        let hints = hints.borrow();
        let block_logslope_layout = logslope_topology
            .materialize_identity(logslope_design.design.clone(), &common_logslope_offset)?;
        block_logslope_layout.validate_for(spec.z.ncols())?;
        let mut cursor = 0usize;
        let rho_time = rho
            .slice(s![cursor..cursor + time_penalties_len])
            .to_owned();
        cursor += time_penalties_len;
        let rho_marginal = rho
            .slice(s![cursor..cursor + marginal_design.penalties.len()])
            .to_owned();
        cursor += marginal_design.penalties.len();
        let rho_logslope = rho
            .slice(s![cursor..cursor + logslope_design.penalties.len()])
            .to_owned();
        cursor += logslope_design.penalties.len();
        let score_warp_active = match flex {
            FlexActivation::On => score_warp_prepared.as_ref(),
            FlexActivation::OffForRigidPilot => None,
        };
        let link_dev_active = match flex {
            FlexActivation::On => link_dev_prepared.as_ref(),
            FlexActivation::OffForRigidPilot => None,
        };
        // The absorbed influence block (#461) is suppressed during the rigid
        // pilot (its residualization derives FROM that pilot); active otherwise.
        let influence_active = match flex {
            FlexActivation::On => influence_absorber_residualized.as_ref(),
            FlexActivation::OffForRigidPilot => None,
        };
        // The warm-start hint `hints.time_beta` is seeded from the rigid
        // pilot's time block (line ~21559). After the pilot's identifiability
        // reduction the stored β can have a *lower* dimension than the raw
        // `design_exit.ncols()` used to build `time_linear_constraints` here
        // (issue #374: with `logslope_formula="1"` the rigid pilot fires,
        // seeds a reduced-width `time_beta`, and feeding it straight into the
        // raw-width projection panicked on an ndarray shape mismatch). Only a
        // hint whose length matches the projection dimension is geometrically
        // meaningful; otherwise fall back to the spec's `initial_beta`, then to
        // the origin inside the projection.
        let time_dim = design_exit.ncols();
        let time_beta_seed = hints
            .time_beta
            .as_ref()
            .filter(|beta| beta.len() == time_dim)
            .or_else(|| {
                time_block_ref
                    .initial_beta
                    .as_ref()
                    .filter(|beta| beta.len() == time_dim)
            });
        let time_beta_hint = if let Some(constraints) = time_linear_constraints.as_ref() {
            Some(project_onto_linear_constraints(
                time_dim,
                constraints,
                time_beta_seed,
            )?)
        } else {
            time_beta_seed.cloned()
        };
        // Same width-guard pattern as `time_beta_seed` above: the cached
        // β-hint is meaningful only when its length matches the current
        // block design width. The hint can outlive a design rebuild (e.g.
        // pilot-time identifiability reduction vs the real fit's reduction,
        // or a κ-probe rematerializing raw-width designs into a slot whose
        // hint was captured at the compiled width); feeding a stale length
        // through to `ParameterBlockSpec` would trip the p_b validation
        // contract with a noisy mid-fit error instead of a clean fall-back
        // to the design's natural cold start.
        let marginal_beta_hint = hints
            .marginal_beta
            .as_ref()
            .filter(|beta| beta.len() == marginal_design.design.ncols())
            .cloned();
        let logslope_beta_hint = hints
            .logslope_beta
            .as_ref()
            .filter(|beta| beta.len() == block_logslope_layout.coefficient_design().ncols())
            .cloned();
        let mut blocks = vec![
            build_time_blockspec(&time_block_ref, &design_exit, rho_time, time_beta_hint),
            build_marginal_blockspec(
                marginal_design,
                &spec.marginal_offset,
                rho_marginal,
                marginal_beta_hint,
            ),
            build_logslope_blockspec(
                logslope_design,
                &block_logslope_layout,
                baseline_slope,
                &spec.logslope_offset,
                rho_logslope,
                logslope_beta_hint,
                Arc::clone(&z),
                score_covariance.clone(),
            )?,
        ];
        if let Some(prepared) = score_warp_active {
            let rho_h = rho
                .slice(s![cursor..cursor + prepared.block.penalties.len()])
                .to_owned();
            cursor += prepared.block.penalties.len();
            blocks.push(build_per_z_score_warp_aux_blockspec(
                prepared,
                rho_h,
                hints.score_warp_beta.clone(),
            )?);
        }
        push_deviation_aux_blockspecs(
            &mut blocks,
            rho,
            &mut cursor,
            None,
            link_dev_active,
            None,
            hints.link_dev_beta.clone(),
        )?;
        // Absorbed Stage-1 influence block (#461): a trailing additive block whose
        // design is the residualized leakage columns `Z̃_infl` and whose single
        // identity penalty `½·λ·‖γ‖²` is REML-learned from the trailing rho slot
        // (seeded at `influence_absorber_log_lambda(n)`). Its
        // gauge priority (130) sits strictly between marginal (150) and logslope
        // (120): the residualization already removes the marginal-aligned
        // component, and the 130 tier makes the canonical-gauge RRQR demote the
        // *logslope* direction (not the absorber) on any shared leakage axis — the
        // discrete realization of `ψ − Π_η[ψ]`. Dropped at predict.
        if let Some(z_tilde) = influence_active {
            let p_i = z_tilde.ncols();
            // The absorber's single learned ridge is the trailing rho slot.
            // It is the last block, so `cursor` is not advanced past it (nothing
            // downstream consumes a further slice).
            let rho_i = rho.slice(s![cursor..cursor + 1]).to_owned();
            let beta_i = hints
                .influence_beta
                .clone()
                .filter(|beta| beta.len() == p_i)
                .unwrap_or_else(|| Array1::<f64>::zeros(p_i));
            blocks.push(ParameterBlockSpec {
                name: "influence_absorber".to_string(),
                design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
                    z_tilde.clone(),
                )),
                offset: Array1::zeros(z_tilde.nrows()),
                penalties: vec![PenaltyMatrix::Dense(Array2::<f64>::eye(p_i))],
                nullspace_dims: vec![0],
                initial_log_lambdas: rho_i,
                initial_beta: Some(beta_i),
                gauge_priority: 130,
                jacobian_callback: None,
                stacked_design: None,
                stacked_offset: None,
            });
        }
        // When timewiggle is active, replace the rigid time and marginal
        // Jacobians with the timewiggle-aware versions.  These compute
        // the full (∂q_r/∂β_t, ∂q_r/∂β_m) chain-rule corrections from
        // the embedded designs + β, without needing a family reference.
        let p_tw = derived_time_wiggle_ncols.unwrap_or(0);
        if p_tw > 0 {
            if let Some(timewiggle) = spec.timewiggle_block.as_ref() {
                let p_m = marginal_design.design.ncols();
                // Densify time designs (already densified earlier in the
                // V+M-exact path; densify again cheaply here — or reuse
                // if the earlier path failed and we are on the raw path).
                let maybe_tw_jac: Option<(
                    Arc<dyn crate::custom_family::BlockEffectiveJacobian>,
                    Arc<dyn crate::custom_family::BlockEffectiveJacobian>,
                )> = (|| {
                    let d_entry = design_entry
                        .try_to_dense_arc("build_blocks::tw_jac::entry")
                        .ok()?;
                    let d_exit = design_exit
                        .try_to_dense_arc("build_blocks::tw_jac::exit")
                        .ok()?;
                    let d_deriv = design_derivative_exit
                        .try_to_dense_arc("build_blocks::tw_jac::deriv")
                        .ok()?;
                    let d_marg = marginal_design
                        .design
                        .try_to_dense_arc("build_blocks::tw_jac::marginal")
                        .ok()?;
                    let knots = timewiggle.knots.clone();
                    let degree = timewiggle.degree;
                    let marginal_offset = Arc::new(spec.marginal_offset.clone());
                    let time_jac = Arc::new(SmsTimewiggleTimeJacobian::new(
                        Arc::clone(&d_entry),
                        Arc::clone(&d_exit),
                        Arc::clone(&d_deriv),
                        Arc::clone(&d_marg),
                        Arc::clone(&offset_entry),
                        Arc::clone(&offset_exit),
                        Arc::clone(&derivative_offset_exit),
                        Arc::clone(&marginal_offset),
                        knots.clone(),
                        degree,
                        p_tw,
                        p_m,
                    ))
                        as Arc<dyn crate::custom_family::BlockEffectiveJacobian>;
                    let marginal_jac = Arc::new(SmsTimewiggleMarginalJacobian::new(
                        d_entry,
                        d_exit,
                        d_deriv,
                        d_marg,
                        Arc::clone(&offset_entry),
                        Arc::clone(&offset_exit),
                        Arc::clone(&derivative_offset_exit),
                        marginal_offset,
                        knots,
                        degree,
                        design_exit.ncols(),
                        p_tw,
                    ))
                        as Arc<dyn crate::custom_family::BlockEffectiveJacobian>;
                    Some((time_jac, marginal_jac))
                })();
                if let Some((time_jac, marginal_jac)) = maybe_tw_jac {
                    blocks[0].jacobian_callback = Some(time_jac);
                    blocks[1].jacobian_callback = Some(marginal_jac);
                }
            }
        }
        Ok(blocks)
    };

    // ── Pilot fit: rigid (zero-penalty) to seed coefficients ────────────
    //
    // The pilot is only a cold-start coefficient initializer. If the caller
    // supplied a warm-start cache session, the outer optimizer will consume
    // that ρ seed and the first real inner solve will immediately overwrite
    // these hints at the cached smoothing point.
    // Running the rigid pilot in that regime is pure latency at large scale
    // (the log shows ~15s for n≈196k), and worse, it seeds β at ρ=0 while the
    // cached outer seed may be far from ρ=0. Do a non-consuming peek so the
    // optimizer still receives the cached entry via `try_load`.
    //
    // The peek must use the same validity criterion as the outer optimizer's
    // cache loader. A poisoned all-boundary checkpoint is not a usable seed:
    // skipping the pilot for such an entry leaves the subsequent cold seed
    // validation without coefficient hints, which is exactly the failure mode
    // this pilot exists to prevent.
    let outer_cache_seed_available = options
        .cache_session
        .as_ref()
        .and_then(|session| session.peek_load_with_source())
        .is_some_and(|loaded| {
            gam_solve::rho_optimizer::cache_entry_would_help_outer(&loaded, setup.rho_dim())
        });
    if outer_cache_seed_available || n < 1_000 {
        let reason = if outer_cache_seed_available {
            "outer-cache-seed-present"
        } else {
            "tiny-fit"
        };
        log::info!(
            "[survival-marginal-slope/pilot] skip reason={} n={} rho_dim={}",
            reason,
            n,
            setup.rho_dim(),
        );
    } else {
        let pilot_started = std::time::Instant::now();
        log::info!(
            "[survival-marginal-slope/pilot] start n={} time_p={} marginal_p={} logslope_p={}",
            n,
            design_exit.ncols(),
            marginal_design.design.ncols(),
            logslope_design.design.ncols(),
        );
        // Pilot ρ has exactly the parametric block sizes — score_warp and
        // link_dev are excluded via FlexActivation::OffForRigidPilot below.
        // Sizing must match build_blocks(... OffForRigidPilot) or the cursor
        // walk inside the closure would slice past the end of the array.
        let rigid_rho = Array1::<f64>::zeros(
            time_penalties_len + marginal_design.penalties.len() + logslope_design.penalties.len(),
        );
        let rigid_blocks = build_blocks(
            &rigid_rho,
            &marginal_design,
            &logslope_design,
            FlexActivation::OffForRigidPilot,
        )?;
        let rigid_family = make_family(
            &marginal_design,
            &logslope_design,
            &initial_hyper_theta,
            FlexActivation::OffForRigidPilot,
        )?;
        let mut pilot_options = options.clone();
        // The pilot is only a warm start. Avoid production covariance assembly
        // and cap inner cycles so a bad seed cannot silently consume minutes
        // before the real outer optimizer starts. Empirically, large-scale
        // survival pilots descend the joint objective by ~5 orders of
        // magnitude in the first 10 cycles and then enter a trust-region-
        // clipped tail; 30 cycles is a budget that catches the descent
        // shoulder without burning into the long tail. At ~0.5s/cycle for
        // a 350k-row LOSO fold that's ~15s — within the "no silent
        // minutes" envelope this cap protects.
        pilot_options.compute_covariance = false;
        pilot_options.inner_max_cycles = pilot_options.inner_max_cycles.min(30);
        match fit_custom_family_fixed_log_lambda_warm_start(
            &rigid_family,
            &rigid_blocks,
            &pilot_options,
        ) {
            Ok((block_beta, converged, cycles)) => {
                // Only install the pilot's β as warm-start hints if the pilot
                // actually reached a KKT certificate. The blockwise inner
                // logger at custom_family.rs:12136 emits the warning
                //   "returning non-converged warm-start iterate and rejecting
                //    this outer REML/LAML evaluation"
                // when its cycle budget is exhausted without convergence; the
                // matching outer-side contract is `nonconverged_outer_eval_result`
                // (custom_family.rs:5993), which surfaces zero gradient and
                // HessianValue::Unavailable so the optimizer backs off. A
                // partial pilot β can still be far from the cold-start optimum
                // (the warning literally exists to signal that), so seeding
                // the real outer optimizer with it can drag the first true
                // inner solve to a degenerate region of (ρ, β)-space from
                // which the analytic envelope gradient is no longer reliable.
                // Discarding the partial β reverts the first real inner solve
                // to a clean cold start at whatever ρ the outer optimizer
                // picks (cached seed or initial_theta), which is the
                // behaviour the warning text already promises.
                if converged {
                    // Pilot only seeds the three parametric blocks. Flex
                    // (score_warp / link_dev) blocks are intentionally absent
                    // under FlexActivation::OffForRigidPilot — there is no
                    // pilot β for them to seed.
                    let mut hints_mut = hints.borrow_mut();
                    if let Some(beta) = block_beta.first() {
                        hints_mut.time_beta = Some(beta.clone());
                    }
                    if let Some(beta) = block_beta.get(1) {
                        hints_mut.marginal_beta = Some(beta.clone());
                    }
                    if let Some(beta) = block_beta.get(2) {
                        hints_mut.logslope_beta = Some(beta.clone());
                    }
                }
                log::info!(
                    "[survival-marginal-slope/pilot] end status={} cycles={} elapsed={:.3}s hints_installed={}",
                    if converged { "converged" } else { "partial" },
                    cycles,
                    pilot_started.elapsed().as_secs_f64(),
                    converged,
                );
            }
            Err(err) => {
                // Pilot audit policy: warn-and-proceed (exploratory).
                //
                // The pilot is a pure warm-start coefficient initialiser — it
                // runs at ρ=0 (no smoothing penalty) with a capped inner-cycle
                // budget solely to seed β hints for the first real inner solve.
                // Rank-deficiency at the pilot stage is a known hazard: the
                // zero-penalty rigid design can expose directions that become
                // identifiable once the outer optimizer selects a non-zero ρ.
                // Raising here would abort the entire fit for a transient
                // structural artifact of the exploration point, not a property
                // of the actual penalised model.
                //
                // Contrast with the outer-inner-fit audit policy (fail-fatal):
                // `fit_custom_family` routes through
                // `canonicalize_for_identifiability`, which returns
                // `CustomFamilyError::IdentifiabilityFailure` on a fatal audit.
                // At the outer fit the full penalty is in play; rank-deficiency
                // there is a genuine model-specification problem that must be
                // surfaced to the caller rather than silently absorbed.
                //
                // In short: pilot tolerates rank-deficiency because it is
                // exploring ρ=0 (a singularity the outer optimizer will never
                // actually accept); outer-inner-fit does not because it operates
                // at the penalised optimum where identifiability is a hard
                // contract.
                log::warn!(
                    "[survival-marginal-slope/pilot] end status=ignored-error elapsed={:.3}s error={}",
                    pilot_started.elapsed().as_secs_f64(),
                    err,
                );
            }
        }
    }

    let marginal_terms = spatial_length_scale_term_indices(&marginalspec_boot);
    let logslope_terms = spatial_length_scale_term_indices(&logslopespec_boot);
    let marginal_has_spatial = !marginal_terms.is_empty();
    let logslope_has_spatial = !logslope_terms.is_empty();
    let analytic_joint_derivatives_available =
        marginal_has_spatial || logslope_has_spatial || setup.log_kappa_dim() == 0;

    if setup.log_kappa_dim() > 0 && !analytic_joint_derivatives_available {
        return Err(
            "exact survival marginal-slope spatial optimization requires analytic joint psi derivatives"
                .to_string(),
        );
    }

    let derivative_probe_started = std::time::Instant::now();
    log::info!(
        "[survival-marginal-slope] initial derivative probe start rho_dim={} log_kappa_dim={}",
        setup.rho_dim(),
        setup.log_kappa_dim(),
    );
    let initial_rho = setup.theta0().slice(s![..setup.rho_dim()]).to_owned();
    let initial_blocks = build_blocks(
        &initial_rho,
        &marginal_design,
        &logslope_design,
        FlexActivation::On,
    )?;
    // Validate the assembled block specs at the construction boundary so any
    // design/penalty width inconsistency surfaces here as a clean typed error
    // string. Without this, the inconsistency would only be
    // caught by the internal `assert_valid_blockspecs` invariant guards inside
    // the capability-query hooks (`outer_hyper_hessian_dense_available`, …)
    // reached from `custom_family_outer_derivatives` below, firing a bare
    // `assert!` panic that PyO3 re-raises as an opaque "panicked inside Rust
    // boundary" GamError instead of an actionable message.
    crate::custom_family::validate_blockspecs(&initial_blocks).map_err(|reason| {
        format!("[survival-marginal-slope] assembled block specs invalid: {reason}")
    })?;
    let initial_family = make_family(
        &marginal_design,
        &logslope_design,
        &initial_hyper_theta,
        FlexActivation::On,
    )?;
    let (joint_gradient, joint_hessian) =
        custom_family_outer_derivatives(&initial_family, &initial_blocks, options);
    let analytic_joint_gradient_available = analytic_joint_derivatives_available
        && matches!(joint_gradient, gam_problem::Derivative::Analytic);
    // Survival marginal-slope now exposes exact coefficient-space and ψ-space
    // Hessian directional derivatives as HyperOperators (see the workspace
    // overrides below). Keep analytic curvature advertised at large scale;
    // the unified REML/LAML planner chooses the matrix-free outer-HVP route for
    // large `(n, p, K)` shapes instead of falling back to first-order BFGS.
    let analytic_joint_hessian_available =
        analytic_joint_derivatives_available && joint_hessian.is_analytic();
    log::info!(
        "[survival-marginal-slope] initial derivative probe end gradient_analytic={} hessian_analytic={} elapsed={:.3}s",
        analytic_joint_gradient_available,
        analytic_joint_hessian_available,
        derivative_probe_started.elapsed().as_secs_f64(),
    );
    let kappa_options_ref: &SpatialLengthScaleOptimizationOptions = kappa_options_effective;
    let hyper_layout_cache = RefCell::new(
        None::<(
            Array1<f64>,
            crate::custom_family::SharedCustomFamilyHyperLayout,
        )>,
    );
    let theta_matches = |left: &Array1<f64>, right: &Array1<f64>| -> bool {
        left.len() == right.len()
            && left
                .iter()
                .zip(right.iter())
                .all(|(lhs, rhs)| lhs.to_bits() == rhs.to_bits())
    };
    let get_hyper_layout = |theta: &Array1<f64>,
                            specs: &[TermCollectionSpec],
                            designs: &[TermCollectionDesign]|
     -> Result<
        crate::custom_family::SharedCustomFamilyHyperLayout,
        String,
    > {
        if let Some((cached_theta, cached_layout)) = hyper_layout_cache.borrow().as_ref()
            && theta_matches(cached_theta, theta)
        {
            return Ok(Arc::clone(cached_layout));
        }

        let mut derivative_blocks = vec![
            Vec::new(),
            if marginal_has_spatial {
                build_block_spatial_psi_derivatives(data, &specs[0], &designs[0])?.ok_or_else(
                    || {
                        "survival marginal-slope: marginal block has spatial terms but spatial psi derivatives are unavailable"
                            .to_string()
                    },
                )?
            } else {
                Vec::new()
            },
            if logslope_has_spatial {
                build_block_spatial_psi_derivatives(data, &specs[1], &designs[1])?.ok_or_else(
                    || {
                        "survival marginal-slope: logslope block has spatial terms but spatial psi derivatives are unavailable"
                            .to_string()
                    },
                )?
            } else {
                Vec::new()
            },
        ];
        if score_warp_runtime.is_some() {
            derivative_blocks.push(Vec::new());
        }
        if link_dev_runtime.is_some() {
            derivative_blocks.push(Vec::new());
        }
        let family_axis_count =
            baseline_axis_count + usize::from(learned_sigma_initial.is_some());
        let family_axes = (0..family_axis_count).collect();
        let hyper_values = theta.slice(s![setup.rho_dim()..]).to_owned();
        let layout = Arc::new(crate::custom_family::CustomFamilyHyperLayout::new(
            derivative_blocks,
            family_axes,
            hyper_values,
        )?);
        hyper_layout_cache.replace(Some((theta.clone(), Arc::clone(&layout))));
        Ok(layout)
    };

    log::info!(
        "[survival-marginal-slope/outer] solve start rho_dim={} log_kappa_dim={} aux_dim={}",
        setup.rho_dim(),
        setup.log_kappa_dim(),
        setup.auxiliary_dim(),
    );

    // Survival marginal-slope is a multi-block family with β-dependent
    // joint Hessian (hazard multipliers depend on current β); the
    // Wood-Fasiolo PSD invariant that justifies EFS fails here, so
    // disable fixed-point at plan time.
    let outer_policy = {
        let psi_dim = setup.theta0().len() - setup.rho_dim();
        initial_family.outer_derivative_policy(&initial_blocks, psi_dim, options)
    };
    let exact_spatial_outer_tol = kappa_options_ref.rel_tol.max(1e-6);
    let solved = optimize_spatial_length_scale_exact_joint(
        data,
        &[marginalspec_boot.clone(), logslopespec_boot.clone()],
        &[marginal_terms.clone(), logslope_terms.clone()],
        kappa_options_ref,
        &setup,
        crate::seeding::SeedRiskProfile::Survival,
        analytic_joint_gradient_available,
        analytic_joint_hessian_available,
        true,
        None,
        outer_policy,
        |theta, specs: &[TermCollectionSpec], designs: &[TermCollectionDesign], provenance| {
            assert_eq!(
                specs.len(),
                designs.len(),
                "survival-marginal-slope outer-inner-fit: specs/designs length mismatch",
            );
            let eval_started = std::time::Instant::now();
            log::info!(
                "[survival-marginal-slope/outer-inner-fit] start theta_dim={}",
                theta.len(),
            );
            let rho = theta.slice(s![..setup.rho_dim()]).to_owned();
            let blocks = build_blocks(
                &rho,
                &designs[0],
                &designs[1],
                FlexActivation::On,
            )?;
            let family = make_family(
                &designs[0],
                &designs[1],
                theta,
                FlexActivation::On,
            )?;
            let fit = match provenance {
                SpatialFitProvenance::NoOuterOptimization => inner_fit(&family, &blocks, options)?,
                SpatialFitProvenance::Certified { outer, mode } => {
                    inner_fit_from_certified_outer(&family, &blocks, options, mode, theta, outer)?
                }
            };
            let mut hints_mut = hints.borrow_mut();
            if let Some(block) = fit.block_states.first() {
                hints_mut.time_beta = Some(block.beta.clone());
            }
            if let Some(block) = fit.block_states.get(1) {
                hints_mut.marginal_beta = Some(block.beta.clone());
            }
            if let Some(block) = fit.block_states.get(2) {
                hints_mut.logslope_beta = Some(block.beta.clone());
            }
            if score_warp_prepared.is_some()
                && let Some(block) = fit.block_states.get(3)
            {
                hints_mut.score_warp_beta = Some(block.beta.clone());
            }
            if link_dev_prepared.is_some() {
                let link_idx = if score_warp_prepared.is_some() { 4 } else { 3 };
                if let Some(block) = fit.block_states.get(link_idx) {
                    hints_mut.link_dev_beta = Some(block.beta.clone());
                }
            }
            log::info!(
                "[survival-marginal-slope/outer-inner-fit] end elapsed={:.3}s inner_cycles={} pirls_status={:?}",
                eval_started.elapsed().as_secs_f64(),
                fit.inner_cycles,
                fit.convergence_evidence().inner_status(),
            );
            Ok(fit)
        },
        |theta,
         specs: &[TermCollectionSpec],
         designs: &[TermCollectionDesign],
         eval_mode,
         row_set: &crate::row_kernel::RowSet| {
            use gam_problem::EvalMode;
            let row_set_rows = match row_set {
                crate::row_kernel::RowSet::All => outer_row_indices(options, n).len(),
                crate::row_kernel::RowSet::Subsample { rows, .. } => rows.len(),
            };
            let eval_started = std::time::Instant::now();
            log::info!(
                "[survival-marginal-slope/outer-eval] start mode={:?} theta_dim={} row_set_rows={}",
                eval_mode,
                theta.len(),
                row_set_rows,
            );
            let rho = theta.slice(s![..setup.rho_dim()]).to_owned();
            let blocks = build_blocks(
                &rho,
                &designs[0],
                &designs[1],
                FlexActivation::On,
            )?;
            if let Some(beta_seed) = pending_beta_seed.borrow_mut().take() {
                let widths: Vec<usize> = blocks.iter().map(|b| b.design.ncols()).collect();
                match CustomFamilyWarmStart::from_cached_beta(&widths, &beta_seed) {
                    Ok(ws) => {
                        exact_warm_start.replace(Some(ws));
                    }
                    Err(e) => {
                        log::warn!(
                            "[SMS] outer ρ-cache β-warm-start rejected: {e}; falling back to cold β"
                        );
                    }
                }
            }
            // Preserve ValueOnly probes and request the Hessian exactly when
            // this realized family advertised analytic joint second-order
            // support.
            let effective_mode = match eval_mode {
                EvalMode::ValueGradientHessian if !analytic_joint_hessian_available => {
                    EvalMode::ValueAndGradient
                }
                other => other,
            };
            let family = make_family(
                &designs[0],
                &designs[1],
                theta,
                FlexActivation::On,
            )?;
            let hyper_layout = get_hyper_layout(theta, specs, designs)?;
            let eval_id = outer_eval_counter.get();
            outer_eval_counter.set(eval_id.wrapping_add(1));
            let tolerance_options =
                joint_hyper_options_for_outer_tolerance(options, exact_spatial_outer_tol);
            let mut outer_options = crate::outer_subsample::exact_outer_options_for_row_set(
                &tolerance_options,
                row_set,
            );
            outer_options.outer_eval_context = Some(crate::custom_family::OuterEvalContext {
                rho: std::sync::Arc::new(rho.clone()),
                eval_id,
                scope: crate::custom_family::EvalScope::OuterDerivative,
            });
            let owned = evaluate_custom_family_joint_hyper_owned_shared(
                &family,
                &blocks,
                &outer_options,
                &rho,
                hyper_layout,
                exact_warm_start.borrow().as_ref(),
                effective_mode,
            )?;
            exact_warm_start.replace(Some(owned.result.warm_start.clone()));
            if !owned.result.inner_converged {
                return Err(
                    "exact survival marginal-slope inner solve did not converge".to_string()
                );
            }
            log::info!(
                "[survival-marginal-slope/outer-eval] end objective={:.6e} mode={:?} elapsed={:.3}s",
                owned.result.objective,
                eval_mode,
                eval_started.elapsed().as_secs_f64(),
            );
            if matches!(eval_mode, EvalMode::ValueGradientHessian)
                && analytic_joint_hessian_available
                && !owned.result.outer_hessian.is_analytic()
            {
                // The outer objective was requested WITH its Hessian, and the
                // family can supply one at a well-conditioned mode — but at THIS
                // ρ/κ it could not (gam#979). The load-bearing reason is a
                // genuinely-indefinite constrained inner mode: it is not a Laplace
                // mode, so no SPD outer-Hessian curvature exists there. That is a
                // property of the surface, NOT an implementation fault, so it must
                // not abort the whole fit. Degrade GRACEFULLY: return the finite
                // value and gradient with the Hessian left `Unavailable`. The
                // outer solver declared `DeclaredHessianForm::Either`, so ARC
                // takes a first-order (gradient / cubic-regularization) step at
                // this ρ and walks OUT of the indefinite region toward a κ where
                // the penalty convexifies the mode — instead of the former fatal
                // "did not return an outer Hessian", which stranded the whole fit
                // the first time ARC's re-seed probe landed on a saddle ρ (the
                // measured survival-marginal-slope n=2500 centers=12 failure,
                // AFTER ARC had already descended 1086.6 → 1081.5). A non-finite
                // objective (a genuinely infeasible mode) still routes through the
                // outer infeasible-on-non-finite-cost guard unchanged.
                log::warn!(
                    "[survival-marginal-slope/outer-eval] no analytic outer Hessian at this ρ \
                     (objective={:.6e}, mode={:?}) — the constrained inner mode is indefinite \
                     (not a Laplace mode); degrading to a first-order (value+gradient) evaluation \
                     so the outer solver steps out of the indefinite region rather than aborting.",
                    owned.result.objective,
                    eval_mode,
                );
            }
            Ok(ExactJointEvaluation {
                objective: owned.result.objective,
                gradient: owned.result.gradient,
                hessian: owned.result.outer_hessian,
                mode: owned.mode,
            })
        },
        |theta,
         specs: &[TermCollectionSpec],
         designs: &[TermCollectionDesign],
         row_set: &crate::row_kernel::RowSet| {
            let eval_started = std::time::Instant::now();
            log::info!(
                "[survival-marginal-slope/outer-efs] start theta_dim={}",
                theta.len(),
            );
            let rho = theta.slice(s![..setup.rho_dim()]).to_owned();
            let blocks = build_blocks(
                &rho,
                &designs[0],
                &designs[1],
                FlexActivation::On,
            )?;
            if let Some(beta_seed) = pending_beta_seed.borrow_mut().take() {
                let widths: Vec<usize> = blocks.iter().map(|b| b.design.ncols()).collect();
                match CustomFamilyWarmStart::from_cached_beta(&widths, &beta_seed) {
                    Ok(ws) => {
                        exact_warm_start.replace(Some(ws));
                    }
                    Err(e) => {
                        log::warn!(
                            "[SMS] outer ρ-cache β-warm-start rejected (efs): {e}; falling back to cold β"
                        );
                    }
                }
            }
            let family = make_family(
                &designs[0],
                &designs[1],
                theta,
                FlexActivation::On,
            )?;
            let hyper_layout = get_hyper_layout(theta, specs, designs)?;
            let eval_id = outer_eval_counter.get();
            outer_eval_counter.set(eval_id.wrapping_add(1));
            let tolerance_options =
                joint_hyper_options_for_outer_tolerance(options, exact_spatial_outer_tol);
            let mut outer_options = crate::outer_subsample::exact_outer_options_for_row_set(
                &tolerance_options,
                row_set,
            );
            outer_options.outer_eval_context = Some(crate::custom_family::OuterEvalContext {
                rho: std::sync::Arc::new(rho.clone()),
                eval_id,
                scope: crate::custom_family::EvalScope::OuterDerivative,
            });
            let owned = evaluate_custom_family_joint_hyper_efs_owned_shared(
                &family,
                &blocks,
                &outer_options,
                &rho,
                hyper_layout,
                exact_warm_start.borrow().as_ref(),
            )?;
            exact_warm_start.replace(Some(owned.result.warm_start.clone()));
            if !owned.result.inner_converged {
                return Err(
                    "exact survival marginal-slope EFS inner solve did not converge".to_string(),
                );
            }
            log::info!(
                "[survival-marginal-slope/outer-efs] end elapsed={:.3}s",
                eval_started.elapsed().as_secs_f64(),
            );
            Ok(ExactJointEfsEvaluation {
                evaluation: owned.result.efs_eval,
                mode: owned.mode,
            })
        },
        crate::marginal_slope_shared::make_beta_seed_validator(&pending_beta_seed),
    );
    // Log the outer-solve outcome on BOTH paths: the inner-solve non-convergence
    // abort (#979/#1040) returns Err before the success log below, so without
    // this the failure stage would be invisible to a `log` backend.
    let solved = match solved {
        Ok(s) => s,
        Err(e) => {
            log::warn!(
                "[survival-marginal-slope/outer] solve FAILED n={n} elapsed={:.3}s reason={e}",
                fit_started.elapsed().as_secs_f64(),
            );
            return Err(e);
        }
    };
    log::info!(
        "[survival-marginal-slope/outer] solve end n={n} elapsed={:.3}s outer_iters={} inner_cycles={} certified",
        fit_started.elapsed().as_secs_f64(),
        solved.fit.outer_iterations,
        solved.fit.inner_cycles,
    );
    let certified_theta = solved
        .certified_outer
        .as_ref()
        .ok_or_else(|| {
            "survival marginal-slope fit completed without a certified joint hyperparameter vector"
                .to_string()
        })?
        .rho();
    let final_sigma = sigma_from_theta(certified_theta)?;
    let (baseline_offset_residuals, baseline_offset_curvatures, final_baseline_config) = {
        let final_family = make_family(
            &solved.designs[0],
            &solved.designs[1],
            certified_theta,
            FlexActivation::On,
        )?;
        let selected_baseline = match (
            &spec.baseline_hyper,
            final_family.family_hyper.baseline_geometry.as_ref(),
        ) {
            (SurvivalMarginalSlopeBaselineHyperSpec::Linear { config }, None) => config.clone(),
            (
                SurvivalMarginalSlopeBaselineHyperSpec::Nonlinear { .. },
                Some(geometry),
            ) => geometry.baseline_config.clone(),
            (SurvivalMarginalSlopeBaselineHyperSpec::Linear { .. }, Some(_)) => {
                return Err(
                    "fixed linear survival marginal-slope baseline unexpectedly realized family coordinates"
                        .to_string(),
                );
            }
            (SurvivalMarginalSlopeBaselineHyperSpec::Nonlinear { .. }, None) => {
                return Err(
                    "learned nonlinear survival marginal-slope baseline lost its certified geometry"
                        .to_string(),
                );
            }
        };
        let (residuals, curvatures) =
            final_family.offset_channel_geometry(&solved.fit.block_states)?;
        (residuals, curvatures, selected_baseline)
    };

    let mut resolved_specs = solved.resolved_specs;
    let designs = solved.designs;
    Ok(SurvivalMarginalSlopeFitResult {
        fit: solved.fit,
        marginalspec_resolved: resolved_specs.remove(0),
        logslopespec_resolved: resolved_specs.remove(0),
        marginal_design: designs[0].clone(),
        logslope_design: designs[1].clone(),
        gaussian_frailty_sd: final_sigma,
        baseline_config: final_baseline_config,
        baseline_slope,
        baseline_offset_residuals,
        baseline_offset_curvatures,
        z_normalization,
        score_covariance: score_covariance.to_dense(),
        time_block_penalties_len: time_penalties_len,
        time_wiggle_knots: spec
            .timewiggle_block
            .as_ref()
            .map(|wiggle| wiggle.knots.clone()),
        time_wiggle_degree: spec.timewiggle_block.as_ref().map(|wiggle| wiggle.degree),
        time_wiggle_ncols: derived_time_wiggle_ncols.unwrap_or(0),
        score_warp_runtime,
        link_dev_runtime,
        influence_absorber_width: influence_absorber_residualized
            .as_ref()
            .map(|z_tilde| z_tilde.ncols()),
        influence_absorber_design: influence_absorber_residualized,
    })
}
