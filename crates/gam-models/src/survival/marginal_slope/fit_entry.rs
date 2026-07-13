//! The public fitting entry point `fit_survival_marginal_slope_terms`.

use super::*;

/// Per-matrix byte budget for the construction-time identifiability preflight.
const PREFLIGHT_MATERIALIZATION_BUDGET_BYTES: usize = 256 * 1024 * 1024;

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
    let initial_sigma = match &spec.frailty {
        FrailtySpec::GaussianShift {
            sigma_fixed: Some(s),
        } => Some(*s),
        FrailtySpec::None => None,
        FrailtySpec::GaussianShift { sigma_fixed: None } | FrailtySpec::HazardMultiplier { .. } => {
            return Err(SurvivalMarginalSlopeError::InvalidInput {
                reason:
                    "internal: validate_spec should have rejected unsupported marginal-slope frailty"
                        .to_string(),
            }
            .into());
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
    let raw_logslope_design_for_layout = logslope_design.design.clone();
    let common_logslope_offset = &spec.logslope_offset + baseline_slope;
    let raw_logslope_layout = logslope_topology.materialize_identity(
        raw_logslope_design_for_layout.clone(),
        &common_logslope_offset,
    )?;
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
    let origin_logslope_beta =
        Array1::<f64>::zeros(raw_logslope_layout.coefficient_design().ncols());
    let origin_slopes =
        raw_logslope_layout.physical_values(spec.z.ncols(), origin_logslope_beta.view())?;

    // Phase-4b parametric identifiability pre-flight (observability-only).
    //
    // Runs `compile_survival_parametric_designs` on the three SMGS
    // parametric blocks (time, marginal, logslope) with a structural
    // identity row Hessian to detect cross-block aliasing in the
    // (q₀, q₁, q′₁, g) row primary state BEFORE the pilot or outer
    // Newton starts. The drops_by_block tuple is logged at INFO so the
    // user sees an immediate, actionable diagnostic when the joint
    // parametric design carries redundant directions — strictly tighter
    // and more structured than the post-construction
    // `audit_identifiability` fatal gate (which sees only the official
    // per-block design, not the full (entry, exit, derivative_exit)
    // triplet that lives in the family primary operator).
    //
    // Why observability-only here (not applied): the family's
    // `evaluate_blockwise_exact_newton` row-Hessian assembly
    // (`syr_row_into_view` / `row_outer_into_view`) asserts that the
    // captured `marginal_design` / `logslope_design` widths equal the
    // workspace slice widths. Threading the compiled (V-transformed)
    // designs through `make_family` and the downstream PIRLS workspace
    // requires width-consistent updates across the >2 000-line
    // `evaluate_blockwise_exact_newton` family of methods that is
    // currently outside this pre-flight's scope. The
    // `canonicalize_for_identifiability` fail-closed gate covers the
    // resulting unsafe-reduce path; this pre-flight gives the user
    // earlier visibility into the same diagnostic. When the family
    // contract is updated to accept compiled designs (a follow-up
    // commit), replace the `log::info!` below with a call to
    // `apply_survival_parametric_compile_to_designs` and re-route
    // `make_family` to the compiled triplet — that is the one-line
    // promotion from observability-only to active reduction.
    if n < 1_000 {
        log::debug!(
            "[smgs phase-4b preflight] skipped for tiny fit n={n}; \
             budget-sensitive tiny survival fits use the raw parametric design"
        );
    } else {
        use crate::survival::marginal_slope::identifiability::{
            SurvivalRowHessian, compile_survival_parametric_designs,
        };
        let n_rows = spec.time_block.design_entry.nrows();
        let preflight = (|| -> Result<(), String> {
            // The preflight densifies five operator-backed designs
            // simultaneously to run the parametric cross-block compile.
            // Without a per-matrix cap, a tensor-product time block at
            // n=320 000 (e.g. 68 age knots × 8 timewiggle knots) materializes
            // to ~1.4 GiB per matrix and OOMs the host before the
            // observability-only diagnostic ever produces a verdict. Cap each
            // matrix at the strict-mode single-materialization budget
            // (`ResourcePolicy::analytic_operator_required`); when any block
            // exceeds it the closure returns `Err` and the surrounding
            // `warn!`-on-fail handler skips the preflight just like any other
            // densification refusal — the downstream
            // `canonicalize_for_identifiability` audit remains the source of
            // truth for the same diagnostic.
            let mut dq0 = spec
                .time_block
                .design_entry
                .try_to_dense_by_chunks_budgeted(
                    "smgs phase-4b preflight time_entry",
                    PREFLIGHT_MATERIALIZATION_BUDGET_BYTES,
                )?;
            let mut dq1 = spec
                .time_block
                .design_exit
                .try_to_dense_by_chunks_budgeted(
                    "smgs phase-4b preflight time_exit",
                    PREFLIGHT_MATERIALIZATION_BUDGET_BYTES,
                )?;
            let mut dqd1 = spec
                .time_block
                .design_derivative_exit
                .try_to_dense_by_chunks_budgeted(
                    "smgs phase-4b preflight time_deriv",
                    PREFLIGHT_MATERIALIZATION_BUDGET_BYTES,
                )?;
            let m_dq = marginal_design.design.try_to_dense_by_chunks_budgeted(
                "smgs phase-4b preflight marginal",
                PREFLIGHT_MATERIALIZATION_BUDGET_BYTES,
            )?;
            let m_dqd1 = ndarray::Array2::<f64>::zeros(m_dq.dim());
            // Channel-aware per-subject Fisher Gram (T8). Pilot primary
            // state at β=0: q0 = offset_entry + marginal_offset, q1 =
            // offset_exit + marginal_offset, qd1 = derivative_offset_exit,
            // g = logslope_offset. The marginal predictor enters BOTH the
            // entry and exit channels (see `row_dynamic_q_values`, which adds
            // `block_states[1].eta` to q0 and q1 alike); at β=0 that predictor
            // is `marginal_offset`. All offsets are available before the inner
            // Newton, so the pilot-H is fully determined at preflight time
            // without waiting for a converged β.
            let mut q0_pf = spec.time_block.offset_entry.clone();
            let mut q1_pf = spec.time_block.offset_exit.clone();
            for i in 0..n_rows {
                q0_pf[i] += spec.marginal_offset[i];
                q1_pf[i] += spec.marginal_offset[i];
            }
            let qd1_pf = spec.time_block.derivative_offset_exit.clone();
            // Replace the zero placeholder timewiggle tail columns with the
            // analytic basis-derived time Jacobian at the β=0 pilot state, so
            // the compiler sees the real time block instead of a structural
            // zero (see `overwrite_timewiggle_time_slots_at_pilot`).
            if let Some(timewiggle) = spec.timewiggle_block.as_ref() {
                overwrite_timewiggle_time_slots_at_pilot(
                    &mut dq0, &mut dq1, &mut dqd1, timewiggle, &q0_pf, &q1_pf, &qd1_pf,
                )?;
            }
            let row_hess = SurvivalRowHessian::from_pilot_primary_state(
                &q0_pf,
                &q1_pf,
                &qd1_pf,
                &origin_slopes,
                &spec.z,
                &score_covariance,
                &spec.weights,
                &spec.event_target,
                spec.derivative_guard,
                probit_scale,
            )?;
            let compiled = compile_survival_parametric_designs(
                dq0,
                dq1,
                dqd1,
                m_dq,
                m_dqd1,
                raw_logslope_layout.clone(),
                &row_hess,
            )?;
            let (dt, dm, dg) = compiled.drops_by_block;
            if dt + dm + dg > 0 {
                log::info!(
                    "[smgs phase-4b preflight] cross-block parametric alias detected: \
                     time_drops={} marginal_drops={} logslope_drops={} \
                     (V_time={}→{}, V_marginal={}→{}, V_logslope={}→{}). \
                     Currently observability-only; canonicalize_for_identifiability \
                     fail-closes downstream if the alias also surfaces in the per-block \
                     audit.",
                    dt,
                    dm,
                    dg,
                    spec.time_block.design_exit.ncols(),
                    compiled.v_time.ncols(),
                    marginal_design.design.ncols(),
                    compiled.v_marginal.ncols(),
                    logslope_design.design.ncols(),
                    compiled.v_logslope.ncols(),
                );
            } else {
                log::debug!(
                    "[smgs phase-4b preflight] parametric joint design is rank-clean: \
                     no cross-block aliasing in (q0, q1, qd1, g) primary state \
                     (raw widths time={} marginal={} logslope={})",
                    spec.time_block.design_exit.ncols(),
                    marginal_design.design.ncols(),
                    logslope_design.design.ncols(),
                );
            }
            Ok(())
        })();
        if let Err(reason) = preflight {
            // Pre-flight is observability-only; an internal error here
            // (e.g. densification budget exceeded) does NOT abort the
            // fit — the downstream audit / fail-closed gate remains
            // the source of truth.
            log::warn!("[smgs phase-4b preflight] skipped: {reason}",);
        }
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
    // Joint design column-rank diagnostic. Builds the dense training-row joint
    // design `[time_block.design_exit | marginal | logslope | score_warp |
    // link_dev]` (with the W-metric residuals applied to the flex bases via
    // `design_at_training_with_residual`) and runs column-pivoted QR with a
    // standard relative tolerance. If the detected rank is below the column
    // count, the joint penalised Hessian has a structural null direction at
    // every β and PIRLS will exhibit the runaway documented above. This fires
    // before any inner solve so the failure surfaces with a precise diagnostic
    // rather than a non-convergence symptom many minutes into the fit.
    //
    // Diagnostic only: rank deficiency at this point is handled gracefully by
    // the canonical-gauge pipeline downstream (`canonicalize_for_identifiability`
    // in `custom_family.rs`, called centrally before the inner Newton solve).
    // That pipeline runs RRQR with `gauge_priority` attribution and converts
    // attributed alias drops into per-block selection matrices `T_i`, then
    // solves on reduced specs and lifts coefficients back via `β_raw = T_i θ`.
    // Aborting here would defeat the canonical reduction. We emit an info-
    // level diagnostic with the (block, ncols, rank) tuple so the rank-deficit
    // is visible in the log without being fatal.
    //
    // The diagnostic streams the W-metric joint Gram over row chunks of the
    // operator-backed block designs (see `joint_training_design_preflight`),
    // so it runs at any n in `O(chunk × p_joint + p_joint²)` memory. The
    // previous implementation densified every block, stacked an `(n,
    // p_joint)` joint matrix, and ran column-pivoted QR plus a thin-SVD over
    // it — multi-GiB of co-resident transients at biobank scale (#979
    // survival construction OOM), guarded by a budget that silently skipped
    // the diagnostic at exactly the scale where it matters. The unweighted
    // RRQR rank pass was deleted along with the densification: the W-metric
    // spectrum is the rank diagnostic PIRLS actually solves under (and the
    // two coincide at uniform sample weights).
    let rank_diagnostic_outcome: Result<(), String> = (|| -> Result<(), String> {
        let score_warp_design = score_warp_prepared
            .as_ref()
            .map(|sw| sw.runtime.design_at_training_with_residual(&z_primary))
            .transpose()?
            .map(|m| DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(m)));
        let link_dev_design = link_dev_prepared
            .as_ref()
            .map(|ld| {
                ld.runtime
                    .design_at_training_with_residual(&cross_block_pilot_eta)
            })
            .transpose()?
            .map(|m| DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(m)));
        let mut segments: Vec<JointPreflightSegment> = Vec::with_capacity(5);
        segments.push(JointPreflightSegment {
            block: JointPreflightBlock::Time,
            columns: spec.time_block.design_exit.clone(),
        });
        segments.push(JointPreflightSegment {
            block: JointPreflightBlock::Marginal,
            columns: marginal_design.design.clone(),
        });
        segments.push(JointPreflightSegment {
            block: JointPreflightBlock::Logslope,
            columns: logslope_design.design.clone(),
        });
        if let Some(m) = score_warp_design {
            segments.push(JointPreflightSegment {
                block: JointPreflightBlock::ScoreWarp,
                columns: m,
            });
        }
        if let Some(m) = link_dev_design {
            segments.push(JointPreflightSegment {
                block: JointPreflightBlock::LinkDev,
                columns: m,
            });
        }
        joint_training_design_preflight(&segments, &spec.weights)
            .map_err(|e| format!("survival-marginal-slope joint preflight failed: {e}"))?;
        Ok(())
    })();
    // Observability-only: a failure inside the rank diagnostic must never
    // abort the fit — the canonical-gauge pipeline downstream is the
    // fail-closed authority on identifiability.
    if let Err(reason) = rank_diagnostic_outcome {
        log::warn!("[survival-marginal-slope joint rank diagnostic] skipped: {reason}");
    }
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
        initial_sigma,
        kappa_options_effective,
    );

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
    let sigma_hint = RefCell::new(initial_sigma);
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

    // Phase-4b V+M-exact active cutover: when the parametric joint
    // design carries cross-block aliasing in the (q₀, q₁, q′₁, g) row
    // primary state, build the channel-aware Gram (K^H, K^S) via
    // `build_primary_grams_gpu_or_cpu`, compile a global T via
    // `compile_from_raw_grams`, and apply it through
    // `apply_compiled_map_to_designs`. The inner Newton then operates
    // on a rank-clean reparameterised joint design; at fit result the
    // joint β is lifted via `T · θ` back to raw width so predict-time
    // consumes raw β unchanged.
    //
    // When no aliasing is detected (all compiled block widths equal
    // their raw widths) the cutover is a no-op: raw designs / penalties
    // propagate forward and `smgs_lift_v` stays None.
    // Recompile-after-first-PIRLS-accept context. Captured inside the
    // cutover branch so we can re-run `compile_survival_parametric_designs_per_term`
    // against a data-adaptive row Hessian built at the converged β, then
    // compare drops_by_block against the structural-H pass. If they differ,
    // the structural compile mis-classified at least one direction (the
    // "pilot-curvature trap") and the user is warned with the diff.
    // The recompile uses `block_states[i].eta` (the converged η for the
    // marginal / logslope blocks) when rebuilding q0/q1/qd1/g at the
    // accepted β. Since η already absorbs the per-row offset
    // (η = Xβ + offset), the recompile does not separately consume
    // `marginal_offset` / `logslope_offset` — they are not carried in
    // this context.
    struct SmgsRecompileAfterAcceptContext {
        dq0: ndarray::Array2<f64>,
        dq1: ndarray::Array2<f64>,
        dqd1: ndarray::Array2<f64>,
        m_dq: ndarray::Array2<f64>,
        m_dqd1: ndarray::Array2<f64>,
        logslope_layout: LogslopeLayout,
        time_partition: Vec<std::ops::Range<usize>>,
        marginal_partition: Vec<std::ops::Range<usize>>,
        logslope_partition: Vec<std::ops::Range<usize>>,
        offset_entry: Array1<f64>,
        offset_exit: Array1<f64>,
        derivative_offset_exit: Array1<f64>,
        z: Array2<f64>,
        covariance: MarginalSlopeCovariance,
        weights: Array1<f64>,
        event: Array1<f64>,
        derivative_guard: f64,
        probit_scale: f64,
        drops_by_block_initial: (usize, usize, usize),
        // #979: true when the cutover engaged the W-orthogonal PARTIAL
        // reduced-logslope reparam (effective Schur Gram) rather than the
        // full per-term compiler. In that case the post-accept recompile —
        // which re-runs the FULL per-term compiler — collapses the logslope
        // channel WHOLESALE by construction, so its drops legitimately differ
        // from the partial structural drops and must NOT be flagged as a
        // pilot-curvature trap (that comparison is apples-to-oranges).
        used_partial_logslope_reduction: bool,
        /// Protect the time block from reduction on recompile — set when the
        /// time block carries a monotone time-wiggle basis (a fixed nonlinear
        /// functional basis that cannot be linearly reparameterised).
        protect_time: bool,
    }
    type SmgsCutoverTuple = (
        gam_linalg::matrix::DesignMatrix,
        gam_linalg::matrix::DesignMatrix,
        gam_linalg::matrix::DesignMatrix,
        gam_terms::smooth::TermCollectionDesign,
        gam_terms::smooth::TermCollectionDesign,
        LogslopeLayout,
        Option<gam_solve::gauge::Gauge>,
        Option<Vec<crate::custom_family::PenaltyMatrix>>,
        Option<Vec<crate::custom_family::PenaltyMatrix>>,
        Option<Vec<crate::custom_family::PenaltyMatrix>>,
        Option<SmgsRecompileAfterAcceptContext>,
    );
    let (
        design_entry,
        design_exit,
        design_derivative_exit,
        marginal_design,
        logslope_design,
        post_cutover_logslope_layout,
        smgs_lift_v,
        time_penalties_vm,
        marginal_penalties_vm,
        logslope_penalties_vm,
        recompile_after_accept,
    ): SmgsCutoverTuple = if n < 1_000 {
        log::debug!(
            "[smgs phase-4b active] skipped for tiny fit n={n}; \
             budget-sensitive tiny survival fits use the raw parametric design"
        );
        (
            spec.time_block.design_entry.clone(),
            spec.time_block.design_exit.clone(),
            spec.time_block.design_derivative_exit.clone(),
            marginal_design.clone(),
            logslope_design.clone(),
            raw_logslope_layout.clone(),
            None,
            None,
            None,
            None,
            None,
        )
    } else {
        use crate::survival::marginal_slope::identifiability::{
            CompiledSurvivalDesignsVMExact, apply_compiled_map_to_designs,
            extract_term_partition_from_penalty_ranges,
        };
        use gam_solve::gauge::Gauge;
        // Recompile context, populated when the closed-form compile
        // succeeds. The post-solve recompile-after-accept hook consumes
        // this to rebuild row Hessians at the converged β.
        let mut recompile_ctx: Option<SmgsRecompileAfterAcceptContext> = None;
        // Try the active cutover via the closed-form compiled-map path.
        // Failure (e.g. densification budget, FullyAliased, linalg error)
        // propagates as Err and skips phase-4b; observability preflight
        // and downstream canonicalize_for_identifiability still gate
        // on the audit.
        let attempt = (|| -> Result<Option<(CompiledSurvivalDesignsVMExact, Gauge)>, String> {
            let n_rows = spec.time_block.design_entry.nrows();
            let p_time = spec.time_block.design_entry.ncols();
            let p_marg = marginal_design.design.ncols();
            let p_log = logslope_design.design.ncols();
            // Single-term partition for the time block: SMGS's time
            // penalty list is over the full time β (one composite
            // smoothness penalty), so a single-term partition is
            // correct here.
            let time_partition: Vec<std::ops::Range<usize>> = std::iter::once(0..p_time).collect();
            let marg_penalty_ranges: Vec<_> = marginal_design
                .penalties
                .iter()
                .map(|p| p.col_range.clone())
                .collect();
            let log_penalty_ranges: Vec<_> = logslope_design
                .penalties
                .iter()
                .map(|p| p.col_range.clone())
                .collect();
            let marginal_partition =
                extract_term_partition_from_penalty_ranges(p_marg, &marg_penalty_ranges);
            let logslope_partition =
                extract_term_partition_from_penalty_ranges(p_log, &log_penalty_ranges);
            // Densify the operator-side designs once. Cap each densification
            // at the strict-mode single-materialization budget so a
            // tensor-product time block at large n does not OOM the host
            // before the closure can return Err — phase-4b is gracefully
            // skipped via the surrounding `warn!`-on-fail match, leaving the
            // downstream `canonicalize_for_identifiability` audit as the
            // gate.
            const ACTIVE_MATERIALIZATION_BUDGET_BYTES: usize = 256 * 1024 * 1024;
            let mut dq0 = spec
                .time_block
                .design_entry
                .try_to_dense_by_chunks_budgeted(
                    "smgs phase-4b active: time_entry",
                    ACTIVE_MATERIALIZATION_BUDGET_BYTES,
                )?;
            let mut dq1 = spec
                .time_block
                .design_exit
                .try_to_dense_by_chunks_budgeted(
                    "smgs phase-4b active: time_exit",
                    ACTIVE_MATERIALIZATION_BUDGET_BYTES,
                )?;
            let mut dqd1 = spec
                .time_block
                .design_derivative_exit
                .try_to_dense_by_chunks_budgeted(
                    "smgs phase-4b active: time_deriv",
                    ACTIVE_MATERIALIZATION_BUDGET_BYTES,
                )?;
            let m_dq = marginal_design.design.try_to_dense_by_chunks_budgeted(
                "smgs phase-4b active: marginal",
                ACTIVE_MATERIALIZATION_BUDGET_BYTES,
            )?;
            let m_dqd1 = ndarray::Array2::<f64>::zeros(m_dq.dim());
            let g_dg = logslope_design.design.try_to_dense_by_chunks_budgeted(
                "smgs phase-4b active: logslope",
                ACTIVE_MATERIALIZATION_BUDGET_BYTES,
            )?;
            // Pilot primary state for the timewiggle Jacobian overwrite
            // below (offset-only β=0 state: q0 = offset_entry +
            // marginal_offset, q1 = offset_exit + marginal_offset, qd1 =
            // derivative_offset_exit, g = logslope_offset). The #808
            // reduction itself uses the RAW stacked design + the
            // operating-point row metric `cross_block_pilot_w`, so it does
            // NOT depend on this pilot primary state; the state is only
            // needed to evaluate the timewiggle basis geometry when the base
            // time basis is disabled (`timewiggle(...)`), so the offset-only
            // state is sufficient and guard-safe.
            let mut q0_pilot = spec.time_block.offset_entry.clone();
            let mut q1_pilot = spec.time_block.offset_exit.clone();
            let qd1_pilot = spec.time_block.derivative_offset_exit.clone();
            let slopes_pilot = origin_slopes.clone();
            for i in 0..n_rows {
                q0_pilot[i] += spec.marginal_offset[i];
                q1_pilot[i] += spec.marginal_offset[i];
            }
            // Replace the zero placeholder timewiggle tail columns with the
            // analytic basis-derived time Jacobian at the pilot state.
            // Without this, the time-channel slots are structurally zero
            // when `timewiggle(...)` disables the base time basis, and the
            // raw stacked design's time block is degenerate.
            if let Some(timewiggle) = spec.timewiggle_block.as_ref() {
                overwrite_timewiggle_time_slots_at_pilot(
                    &mut dq0, &mut dq1, &mut dqd1, timewiggle, &q0_pilot, &q1_pilot, &qd1_pilot,
                )?;
            }

            // Closed-form Gram path on the RAW STACKED design (#808).
            //
            // History: the 4-channel `build_primary_grams_gpu_or_cpu` view
            // (marginal→q0/q1, logslope→g) has a structural Gram that is
            // block-diagonal *by channel*, so marginal⊥logslope structurally
            // and the overlap is invisible (build-1, no drops). The η₁
            // row-Jacobian view (build-2) row-scales the SHARED matern basis
            // by DIFFERENT per-row factors (marginal: c(g); logslope:
            // q1·c1(g)+s_f·z) which are NOT proportional across rows, so it
            // *breaks* the raw collinearity and the Gram comes back FULL RANK
            // (DIAG: W-rank=26/26, alias_dirs=0, despite g_pilot moving to
            // [0.31,0.54]) — also no drops.
            //
            // The alias is a collinearity of the RAW columns: marginal and
            // logslope share the same `matern(PC1,PC2,PC3)` basis evaluated on
            // the same PCs, so the raw stacked design `[time_exit | marginal |
            // logslope]` is genuinely W-rank-deficient (the preflight,
            // `joint_training_design_preflight`, measures exactly this: rank
            // 19/26, 7 alias dirs, dominant cols logslope[0,3] +
            // marginal[1,2,4,5,6]). Detect + reduce in THAT metric: build the
            // Gram on the raw stacked design weighted by the operating-point
            // IRLS row metric `cross_block_pilot_w` (the metric the inner
            // penalised Hessian's near-singularity, cond≈5.8e6, actually
            // tracks; reduces to the preflight's unweighted SVD when weights
            // are uniform and the pilot is flat). `compile_from_raw_grams`
            // then resolves the overlap with cross-block carry (R terms —
            // keep time+marginal high-priority, reparameterise logslope as the
            // W-orthogonal complement; NOT the falsified v2 whole-block
            // deletion). Sound: the raw marginal≈logslope collinearity is a
            // genuine confound (same PC-surface direction represented in both
            // the mean and the log-slope channel; the inner cannot separate
            // them → near-singular H_pen), and cross-block carry is the
            // standard identifiability resolution, here at the operating-point
            // W rather than at β=0.
            {
                use gam_identifiability::families::compiler::{
                    BlockOrder as IdBlockOrder, compile_from_raw_grams_protected,
                };
                let closed_form = (|| -> Result<
                    Option<(
                        gam_identifiability::families::compiler::CompiledMap,
                        (usize, usize, usize),
                        // #979: used_partial_logslope_reduction (see struct doc)
                        bool,
                    )>,
                    String,
                > {
                    let p_total_raw = p_time + p_marg + p_log;
                    let raw_ranges = vec![
                        0..p_time,
                        p_time..(p_time + p_marg),
                        (p_time + p_marg)..p_total_raw,
                    ];
                    if cross_block_pilot_w.len() != n_rows {
                        return Err(format!(
                            "raw-stack Gram: cross_block_pilot_w len {} != n_rows {}",
                            cross_block_pilot_w.len(),
                            n_rows
                        ));
                    }
                    // Raw stacked exit-channel design `[time_exit | marginal |
                    // logslope]` — the same column layout the preflight SVDs.
                    // `dq1` is the time exit design (overwrite_timewiggle already
                    // filled its analytic tail at the pilot state above).
                    let mut j_raw = ndarray::Array2::<f64>::zeros((n_rows, p_total_raw));
                    for i in 0..n_rows {
                        for j in 0..p_time {
                            j_raw[[i, j]] = dq1[[i, j]];
                        }
                        for j in 0..p_marg {
                            j_raw[[i, p_time + j]] = m_dq[[i, j]];
                        }
                        for j in 0..p_log {
                            j_raw[[i, p_time + p_marg + j]] = g_dg[[i, j]];
                        }
                    }
                    // K^S = Xᵀ X (structural — sees the raw marginal≈logslope
                    // collinearity), K^H = Xᵀ diag(w) X (operating-point W metric).
                    let gram_struct = gam_linalg::faer_ndarray::fast_ata(&j_raw);
                    let gram_h = fast_xt_diag_x(&j_raw, &cross_block_pilot_w);
                    // #808 diagnostic: W-metric thin-SVD of the raw stacked design
                    // (mirrors `joint_training_design_preflight`) so we can see
                    // directly whether the reduction metric is rank-deficient (and
                    // by how much) before compile runs.
                    if log::log_enabled!(log::Level::Info) {
                        use gam_linalg::faer_ndarray::FaerSvd;
                        let mut jw = j_raw.clone();
                        for i in 0..n_rows {
                            let s = cross_block_pilot_w[i].max(0.0).sqrt();
                            for j in 0..p_total_raw {
                                jw[[i, j]] *= s;
                            }
                        }
                        if let Ok((_u, sigma, _vt)) = jw.svd(false, false) {
                            let smax = sigma.iter().copied().fold(0.0_f64, f64::max);
                            let tol_dbg = smax
                                * (n_rows.max(p_total_raw) as f64)
                                * 16.0
                                * f64::EPSILON;
                            let n_alias = sigma.iter().filter(|&&s| s <= tol_dbg).count();
                            let smin = sigma.iter().copied().fold(f64::INFINITY, f64::min);
                            let g_range = {
                                let mut lo = f64::INFINITY;
                                let mut hi = f64::NEG_INFINITY;
                                for &g in slopes_pilot.iter() {
                                    lo = lo.min(g);
                                    hi = hi.max(g);
                                }
                                (lo, hi)
                            };
                            log::info!(
                                "[smgs phase-4b rawstack-gram DIAG] sigma_max={smax:.4e} sigma_min={smin:.4e} \
                                 tol={tol_dbg:.4e} W-rank={}/{} alias_dirs={n_alias} g_pilot=[{:.3e},{:.3e}]",
                                p_total_raw - n_alias,
                                p_total_raw,
                                g_range.0,
                                g_range.1,
                            );
                        }
                    }
                    // The monotone time-wiggle time block is a fixed nonlinear
                    // functional basis, not a plain linear design: its
                    // `SmsTimewiggleTimeJacobian` recomputes the full-width
                    // wiggle basis on every evaluation and cannot be expressed
                    // on a linearly reduced time design. Protect it from the
                    // compiled-map reduction so it stays at raw width (its own
                    // wiggle penalty nullspace regularises its conditioning);
                    // marginal/logslope still reduce against the full time
                    // anchor. Without this, a time reduction (11→9 here) leaves
                    // the Jacobian writing `p_tw` wiggle columns into a narrower
                    // `p_time`-wide buffer — an out-of-bounds panic.
                    let time_is_timewiggle = spec.timewiggle_block.is_some();
                    let map = compile_from_raw_grams_protected(
                        &gram_h,
                        &gram_struct,
                        &raw_ranges,
                        &[
                            IdBlockOrder::Time,
                            IdBlockOrder::Marginal,
                            IdBlockOrder::Logslope,
                        ],
                        &[time_is_timewiggle, false, false],
                    )
                    .map_err(|e| format!("compile_from_raw_grams: {e}"))?;
                    if map.raw_from_compiled.shape()[0] != p_total_raw {
                        return Err(format!(
                            "T raw width {} != expected {}",
                            map.raw_from_compiled.shape()[0],
                            p_total_raw
                        ));
                    }
                    if !map.raw_from_compiled.iter().all(|v| v.is_finite()) {
                        return Err("T contains non-finite entries".to_string());
                    }
                    let w_time = map.compiled_block_ranges[0].len();
                    let w_marg = map.compiled_block_ranges[1].len();
                    let w_log = map.compiled_block_ranges[2].len();
                    // #808 root guard: rawstack reduction is only a valid
                    // identifiability cleanup when it preserves the physical
                    // model channels. Clustered-PC SMGS can make the raw
                    // marginal/logslope columns identical even though the full
                    // nonlinear η-Jacobian still distinguishes them. Applying
                    // the map in that case zeroes the entire lower-priority
                    // logslope block and deletes the model's slope channel,
                    // turning a conditioning problem into a misspecified fit.
                    //
                    // Keep non-destructive partial reductions: they remove
                    // redundant raw coordinates while retaining at least one
                    // degree of freedom in each required channel. Reject only
                    // maps that collapse a required channel to zero width.
                    if let Some(channel) = smgs_deleted_required_channel_reason(
                        p_time, p_marg, p_log, w_time, w_marg, w_log,
                    ) {
                        // #741: the η₁-only rawstack W-metric collapsed a whole
                        // required channel. That metric is only the η₁-channel
                        // row curvature; the true survival row Hessian is 4×4 in
                        // (q₀,q₁,qd₁,g) and chains DIFFERENTLY into each block, so
                        // marginal/logslope that look identical in η₁ are kept
                        // distinct by the full driver. Before falling back to the
                        // unreduced (rank-deficient) raw design, retry the
                        // reduction with the full row-Hessian per-term compiler.
                        // If it preserves every required channel, the η₁ collapse
                        // was a FALSE alias — emit its CompiledMap so Newton runs
                        // in the correct identifiable quotient (the closed-form
                        // fast path engages). Only when the full row Hessian ALSO
                        // deletes the channel is the alias real → unreduced design.
                        use crate::survival::marginal_slope::identifiability::{
                            SurvivalRowHessian, compile_survival_parametric_designs_per_term,
                            compiled_map_from_per_term,
                        };
                        let full_row_hess = (|| -> Result<
                            Option<(
                                gam_identifiability::families::compiler::CompiledMap,
                                (usize, usize, usize),
                                // #979: used_partial_logslope_reduction (see struct doc)
                                bool,
                            )>,
                            String,
                        > {
                            let row_hess = SurvivalRowHessian::from_pilot_primary_state(
                                &q0_pilot,
                                &q1_pilot,
                                &qd1_pilot,
                                &slopes_pilot,
                                &spec.z,
                                &score_covariance,
                                &spec.weights,
                                &spec.event_target,
                                derivative_guard,
                                probit_scale,
                            )?;
                            let per_term = compile_survival_parametric_designs_per_term(
                                dq0.clone(),
                                dq1.clone(),
                                dqd1.clone(),
                                &time_partition,
                                m_dq.clone(),
                                m_dqd1.clone(),
                                &marginal_partition,
                                raw_logslope_layout.clone(),
                                &logslope_partition,
                                &row_hess,
                                spec.timewiggle_block.is_some(),
                            )?;
                            let map = compiled_map_from_per_term(&per_term);
                            let fw_time = map.compiled_block_ranges[0].len();
                            let fw_marg = map.compiled_block_ranges[1].len();
                            let fw_log = map.compiled_block_ranges[2].len();
                            if let Some(real) = smgs_deleted_required_channel_reason(
                                p_time, p_marg, p_log, fw_time, fw_marg, fw_log,
                            ) {
                                // #979 residual phantom-null path. The full 4×4
                                // row-Hessian quotient ALSO collapses a required
                                // channel: the effective metric is genuinely
                                // rank-deficient here. Historically this fell back
                                // to the UNREDUCED design + Jeffreys, leaving a
                                // quadratically-flat near-null direction in the
                                // joint penalized Hessian — the inner solve could
                                // not certify stationarity, so the fit could not
                                // converge and reported non-stationarity honestly.
                                //
                                // Instead, MEASURE before deciding. Assemble the
                                // joint penalized Hessian M = JᵀHJ + S and the
                                // joint score g at the pilot, eigendecompose, and
                                // for every near-null direction v measure the
                                // gradient residual r = |vᵀg|. Accept the
                                // channel-collapsing reduction (projecting that
                                // direction away) ONLY when every near-null
                                // direction is an empirical phantom — r ≤ λ·step,
                                // i.e. the optimizer would converge it in under one
                                // trust step, so projecting it changes neither the
                                // optimum nor hides non-stationarity. If any
                                // near-null direction carries real non-stationarity
                                // (r > λ·step, e.g. a near-separating pull), or the
                                // collapsed channel is the spatial TIME block, we
                                // refuse to project and keep the conservative
                                // unreduced + Jeffreys fallback (non-stationarity
                                // is then reported honestly). This is the gate that prevents the
                                // reward-hacking failure mode of silently deleting
                                // a direction the model genuinely needs — exactly
                                // the regression a naive nullspace-shrink caused on
                                // the n ≥ 1000 spatial path.
                                //
                                // #979 ROOT-CAUSE FIX (preferred over the gate
                                // below): when the COLLAPSED channel is logslope,
                                // first try a W-orthogonal PARTIAL reduced-logslope
                                // reparam — the proven-correct BMS construction
                                // (`bms::block_specs::reduced_logslope_transform_effective`)
                                // ported into survival's per-row 4×4 Hessian metric
                                // (`survival_reduced_logslope_transform_effective`).
                                // The full per-term compiler deletes the WHOLE
                                // logslope channel because its priority-ordered RRQR
                                // attributes every shared marginal↔logslope direction
                                // to the lowest-priority logslope block. The effective
                                // Schur Gram instead removes from logslope ONLY the
                                // directions W-explained by the marginal span, KEEPS
                                // the `r` surviving directions, and leaves marginal/
                                // time untouched. The joint penalised Hessian
                                // M = JᵀHJ + S is then full-rank BY CONSTRUCTION — the
                                // 2e14 marginal↔logslope phantom null is gone, the
                                // inner joint-Newton certifies on its own, so the
                                // fit converges without any backstop.
                                // Only the logslope confound is eligible here (the
                                // spatial time block is protected by the gate below).
                                if real == "logslope" {
                                    use crate::survival::marginal_slope::identifiability::{
                                        survival_block_diagonal_logslope_map,
                                        survival_reduced_logslope_transform_effective,
                                    };
                                    use crate::bms::block_specs::ReducedLogslopeOutcome;
                                    match survival_reduced_logslope_transform_effective(
                                        m_dq.view(),
                                        &raw_logslope_layout,
                                        &row_hess,
                                    ) {
                                        Ok(ReducedLogslopeOutcome::Reduced(t_log)) => {
                                            let wl = t_log.ncols();
                                            let bd_map = survival_block_diagonal_logslope_map(
                                                p_time, p_marg, &t_log,
                                            );
                                            log::info!(
                                                "[smgs phase-4b compiled-map] #979: full row-Hessian \
                                                 collapses the logslope channel ({p_log}→0), but the \
                                                 W-orthogonal effective Schur Gram keeps {wl}/{p_log} \
                                                 surviving logslope directions; engaging the BMS-style \
                                                 PARTIAL reduced-logslope reparam (marginal/time pass \
                                                 through unchanged) so the joint penalised Hessian is \
                                                 full-rank by construction — phantom null removed, \
                                                 inner solve certifies on its own",
                                            );
                                            return Ok(Some((bd_map, (p_time, p_marg, wl), true)));
                                        }
                                        Ok(ReducedLogslopeOutcome::FullRank) => {
                                            // No effective confound to remove; fall
                                            // through to the measured-phantom gate.
                                        }
                                        Ok(ReducedLogslopeOutcome::FullyConfounded) => {
                                            // The ENTIRE effective logslope image is
                                            // W-explained by the marginal span — the
                                            // block is unidentified (#2245 finding 45
                                            // sibling). Deleting the whole channel is
                                            // the deliberate handling here: the gate
                                            // below performs exactly that projection,
                                            // now reached explicitly rather than via a
                                            // signal shared with the full-rank case.
                                            log::info!(
                                                "[smgs phase-4b compiled-map] #979: the effective \
                                                 Schur Gram keeps 0/{p_log} logslope directions — \
                                                 the block is fully confounded with the marginal \
                                                 surface; deferring to the measured-phantom gate \
                                                 to delete the unidentified channel",
                                            );
                                        }
                                        Err(reason) => {
                                            log::warn!(
                                                "[smgs phase-4b compiled-map] #979 partial \
                                                 reduced-logslope reparam unavailable ({reason}); \
                                                 falling back to the measured-phantom gate",
                                            );
                                        }
                                    }
                                }
                                let gate = (|| -> Result<bool, String> {
                                    // Protect the spatial/time path: only the
                                    // marginal↔logslope confound is eligible for
                                    // measured projection here.
                                    if real == "time" {
                                        return Ok(false);
                                    }
                                    let time_pen = dense_block_penalty_from_dense_list(
                                        &spec.time_block.penalties,
                                        p_time,
                                    )?;
                                    let marg_pen = dense_block_penalty_from_blockwise(
                                        &marginal_design.penalties,
                                        p_marg,
                                    )?;
                                    let log_pen = dense_block_penalty_from_blockwise(
                                        &logslope_design.penalties,
                                        p_log,
                                    )?;
                                    let s_total = assemble_unit_block_penalty(
                                        p_time, p_marg, p_log, &time_pen, &marg_pen, &log_pen,
                                    )?;
                                    let report = survival_kkt_refusal_report_at_pilot(
                                        SurvivalEffectiveDesigns {
                                            dq0: &dq0,
                                            dq1: &dq1,
                                            dqd1: &dqd1,
                                            m_dq: &m_dq,
                                            m_dqd1: &m_dqd1,
                                            logslope_layout: &raw_logslope_layout,
                                        },
                                        &row_hess,
                                        SurvivalPilotRows {
                                            q0: &q0_pilot,
                                            q1: &q1_pilot,
                                            qd1: &qd1_pilot,
                                            slopes: &slopes_pilot,
                                            z: &spec.z,
                                            covariance: &score_covariance,
                                            weights: &spec.weights,
                                            event: &spec.event_target,
                                        },
                                        SurvivalLinkParams {
                                            derivative_guard,
                                            probit_scale,
                                        },
                                        &s_total,
                                        KKT_PHANTOM_TRUST_RADIUS,
                                    )?;
                                    log::info!(
                                        "[smgs phase-4b kkt-refusal] channel={real} {}",
                                        report.summary(),
                                    );
                                    Ok(report.all_near_null_are_phantom())
                                })();
                                match gate {
                                    Ok(true) => {
                                        log::info!(
                                            "[smgs phase-4b compiled-map] #979: full row-Hessian \
                                             collapses channel {real} (time {p_time}→{fw_time}, \
                                             marginal {p_marg}→{fw_marg}, logslope {p_log}→{fw_log}), \
                                             but the joint penalized Hessian's near-null \
                                             direction(s) are MEASURED phantoms (gradient residual \
                                             ≤ λ·step); engaging the channel-reduced quotient so \
                                             the phantom null direction is projected out — \
                                             inner solve certifies on its own",
                                        );
                                        Ok(Some((map, (fw_time, fw_marg, fw_log), false)))
                                    }
                                    Ok(false) => {
                                        log::warn!(
                                            "[smgs phase-4b compiled-map] full row-Hessian compile \
                                             also deletes channel {real} (time {p_time}→{fw_time}, \
                                             marginal {p_marg}→{fw_marg}, logslope {p_log}→{fw_log}); \
                                             the near-null direction carries real non-stationarity \
                                             (or is the spatial time block) — refusing to project; \
                                             using the unreduced design and leaving the near-null \
                                             direction to Jeffreys conditioning",
                                        );
                                        Ok(None)
                                    }
                                    Err(reason) => {
                                        log::warn!(
                                            "[smgs phase-4b compiled-map] KKT-refusal measurement \
                                             failed ({reason}); conservatively using the unreduced \
                                             design for channel {real}",
                                        );
                                        Ok(None)
                                    }
                                }
                            } else {
                                log::info!(
                                    "[smgs phase-4b compiled-map] #741: η₁-only metric falsely \
                                     collapsed channel {channel}; full 4×4 row-Hessian quotient \
                                     keeps all channels (time {p_time}→{fw_time}, \
                                     marginal {p_marg}→{fw_marg}, logslope {p_log}→{fw_log}); \
                                     engaging closed-form fast path on the correct quotient",
                                );
                                Ok(Some((map, (fw_time, fw_marg, fw_log), false)))
                            }
                        })();
                        match full_row_hess {
                            Ok(some) => Ok(some),
                            Err(reason) => {
                                log::warn!(
                                    "[smgs phase-4b compiled-map] full row-Hessian retry failed \
                                     ({reason}); rawstack metric collapsed channel {channel} — \
                                     using the unreduced design and leaving the near-null \
                                     direction to Jeffreys conditioning",
                                );
                                Ok(None)
                            }
                        }
                    } else {
                        Ok(Some((map, (w_time, w_marg, w_log), false)))
                    }
                })();
                match closed_form {
                    Ok(Some((map, (wt, wm, wl), used_partial_logslope_reduction))) => {
                        let drops = (
                            p_time.saturating_sub(wt),
                            p_marg.saturating_sub(wm),
                            p_log.saturating_sub(wl),
                        );
                        // Populate the post-accept recompile context.
                        // The recompile hook rebuilds from the densified
                        // matrices at converged β; the initial drops field
                        // is purely diagnostic.
                        recompile_ctx = Some(SmgsRecompileAfterAcceptContext {
                            dq0: dq0.clone(),
                            dq1: dq1.clone(),
                            dqd1: dqd1.clone(),
                            m_dq: m_dq.clone(),
                            m_dqd1: m_dqd1.clone(),
                            logslope_layout: raw_logslope_layout.clone(),
                            time_partition: time_partition.clone(),
                            marginal_partition: marginal_partition.clone(),
                            logslope_partition: logslope_partition.clone(),
                            offset_entry: spec.time_block.offset_entry.clone(),
                            offset_exit: spec.time_block.offset_exit.clone(),
                            derivative_offset_exit: spec.time_block.derivative_offset_exit.clone(),
                            z: spec.z.clone(),
                            covariance: score_covariance.clone(),
                            weights: spec.weights.clone(),
                            event: spec.event_target.clone(),
                            derivative_guard,
                            probit_scale,
                            drops_by_block_initial: drops,
                            used_partial_logslope_reduction,
                            protect_time: spec.timewiggle_block.is_some(),
                        });
                        if drops.0 + drops.1 + drops.2 == 0 {
                            log::info!(
                                "[smgs phase-4b compiled-map] compile_from_raw_grams ok with no drops \
                                 (time {p_time}→{wt}, marginal {p_marg}→{wm}, logslope {p_log}→{wl}); \
                                 production path = compiled_map, skipping apply"
                            );
                            return Ok(None);
                        }
                        log::info!(
                            "[smgs phase-4b compiled-map] applying CompiledMap T: \
                             time {p_time}→{wt}, marginal {p_marg}→{wm}, logslope {p_log}→{wl} \
                             (drops time={}, marginal={}, logslope={}); \
                             production path = compiled_map",
                            drops.0,
                            drops.1,
                            drops.2,
                        );
                        let time_pens_bw: Vec<gam_terms::smooth::BlockwisePenalty> = spec
                            .time_block
                            .penalties
                            .iter()
                            .map(|p| gam_terms::smooth::BlockwisePenalty::new(0..p_time, p.clone()))
                            .collect();
                        let applied: CompiledSurvivalDesignsVMExact =
                            apply_compiled_map_to_designs(
                                &map,
                                spec.time_block.design_entry.clone(),
                                spec.time_block.design_exit.clone(),
                                spec.time_block.design_derivative_exit.clone(),
                                marginal_design.design.clone(),
                                logslope_design.design.clone(),
                                &time_pens_bw,
                                &marginal_design.penalties,
                                &logslope_design.penalties,
                            )?;
                        let ordering = [
                            IdBlockOrder::Time,
                            IdBlockOrder::Marginal,
                            IdBlockOrder::Logslope,
                        ];
                        let lift = Gauge::from_compiled_map(&map, &ordering);
                        return Ok(Some((applied, lift)));
                    }
                    Ok(None) => {
                        return Ok(None);
                    }
                    Err(reason) => {
                        return Err(format!("closed-form path unavailable: {reason}"));
                    }
                }
            }
        })();
        match attempt {
            Ok(Some((applied, lift))) => {
                // V+M-exact compiled .design swapped into clones of the
                // raw TermCollectionDesigns. The TermCollectionDesign's
                // .penalties field stays raw (Vec<BlockwisePenalty>) for
                // predict-time consumers; the V+M-exact full-width
                // pulled-back penalties travel via the side bindings
                // `*_penalties_vm` and are wired into the per-block
                // `ParameterBlockSpec.penalties` inside `build_blocks`.
                //
                // Other TermCollectionDesign metadata stays at raw width
                // — it's consumed post-fit, by which point β has been
                // lifted back to raw via `T · θ`.
                let mut marg_out = marginal_design.clone();
                marg_out.design = applied.marginal_design;
                let mut log_out = logslope_design.clone();
                let compiled_logslope_layout = logslope_topology.materialize(
                    raw_logslope_design_for_layout.clone(),
                    applied.logslope_current_from_raw.clone(),
                    &common_logslope_offset,
                )?;
                log_out.design = compiled_logslope_layout.coefficient_design().clone();
                (
                    applied.time_design_entry,
                    applied.time_design_exit,
                    applied.time_design_derivative_exit,
                    marg_out,
                    log_out,
                    compiled_logslope_layout,
                    Some(lift),
                    Some(applied.time_penalties),
                    Some(applied.marginal_penalties),
                    Some(applied.logslope_penalties),
                    recompile_ctx,
                )
            }
            Ok(None) => (
                spec.time_block.design_entry.clone(),
                spec.time_block.design_exit.clone(),
                spec.time_block.design_derivative_exit.clone(),
                marginal_design.clone(),
                logslope_design.clone(),
                raw_logslope_layout.clone(),
                None,
                None,
                None,
                None,
                recompile_ctx,
            ),
            Err(reason) => return Err(format!("[smgs phase-4b active] {reason}")),
        }
    };
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
    // FlexActivation::OffForRigidPilot forces the rigid warm-start to construct
    // a family with no score_warp / link_dev runtimes and no flex blocks. That
    // is the only way to guarantee the pilot does not enter the survival flex
    // exact-Joint-Newton path. A boolean here would be too easy to flip
    // accidentally; the named enum makes the intent and audit obvious at every
    // call site.
    let make_family = |marginal_design: &TermCollectionDesign,
                       logslope_design: &TermCollectionDesign,
                       sigma: Option<f64>,
                       flex: FlexActivation,
                       coords: BlockDesignCoords|
     -> Result<SurvivalMarginalSlopeFamily, String> {
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
        let logslope_layout = match coords {
            BlockDesignCoords::PostCutover => post_cutover_logslope_layout.clone(),
            BlockDesignCoords::RematerializedRaw => logslope_topology
                .materialize_identity(logslope_design.design.clone(), &common_logslope_offset)?,
        };
        logslope_layout.validate_for(spec.z.ncols())?;
        Ok(SurvivalMarginalSlopeFamily {
            n,
            event: Arc::clone(&event),
            weights: Arc::clone(&weights),
            z: Arc::clone(&z),
            score_covariance: score_covariance.clone(),
            gaussian_frailty_sd: sigma,
            derivative_guard,
            design_entry: design_entry.clone(),
            design_exit: design_exit.clone(),
            design_derivative_exit: design_derivative_exit.clone(),
            offset_entry: Arc::clone(&offset_entry),
            offset_exit: Arc::clone(&offset_exit),
            derivative_offset_exit: Arc::clone(&derivative_offset_exit),
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
                        flex: FlexActivation,
                        coords: BlockDesignCoords|
     -> Result<Vec<ParameterBlockSpec>, String> {
        let hints = hints.borrow();
        let block_logslope_layout = match coords {
            BlockDesignCoords::PostCutover => post_cutover_logslope_layout.clone(),
            BlockDesignCoords::RematerializedRaw => logslope_topology
                .materialize_identity(logslope_design.design.clone(), &common_logslope_offset)?,
        };
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
        // V+M-exact cutover: when the active cutover fired, the
        // `*_penalties_vm` side bindings carry per-block-width Dense
        // penalty matrices pulled back through each block's OWN diagonal
        // reparameterisation V_b (V_bᵀ S_b V_b), sized
        // `w_b_compiled × w_b_compiled`. They substitute for each block's
        // raw-width Blockwise penalties so the inner solver sees the exact
        // compiled-coord penalties. The penalty count is invariant under the
        // pullback so cursor accounting based on `design.penalties.len()` (raw
        // widths) still matches. The cross-block residualisation R_{a→b} is
        // carried by the residualised compiled *design* columns, not the
        // penalty, so each block's penalty stays per-block-width — matching the
        // `ParameterBlockSpec` p_b × p_b validation contract.
        //
        // These pulled-back penalties are valid ONLY against the compiled
        // designs they were derived from. The `coords` tag — set by each call
        // site, not inferred from a width coincidence — says whether the
        // designs handed to this call ARE those compiled designs:
        //
        //   * `PostCutover`: the construction-site designs the `_vm` were
        //     pulled back through. Install them, and assert width agreement —
        //     a mismatch here means the compiled designs and compiled penalties
        //     have desynced (a construction-site wiring bug), which we surface
        //     loudly rather than letting `validate_blockspecs` reject it later
        //     with a less actionable message.
        //   * `RematerializedRaw`: the κ-probe re-materialises *raw*-width
        //     marginal/logslope designs from the boot specs and routes them
        //     here. The raw design-derived penalties already installed by
        //     `build_*_blockspec` are authoritative; installing the compiled
        //     `_vm` here is the #788 shape mismatch (and, when widths happen to
        //     coincide with no column drop but `V≠I`, a silent `Vᵀ S V`-on-raw
        //     corruption). Keep the raw penalties.
        if coords == BlockDesignCoords::PostCutover {
            for (block_idx, pens_vm) in [
                (0usize, &time_penalties_vm),
                (1, &marginal_penalties_vm),
                (2, &logslope_penalties_vm),
            ] {
                if let Some(pens) = pens_vm {
                    let w = blocks[block_idx].design.ncols();
                    if !pens.iter().all(|p| p.shape() == (w, w)) {
                        return Err(format!(
                            "survival marginal-slope: compiled V+M penalty/design width desync at \
                             block {block_idx} (compiled design width {w}, penalty shapes {:?}); \
                             the post-cutover compiled designs and `*_penalties_vm` must agree",
                            pens.iter().map(|p| p.shape()).collect::<Vec<_>>()
                        ));
                    }
                    blocks[block_idx].penalties = pens.clone();
                }
            }
        }
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
            // Construction-site designs: the post-cutover compiled marginal/
            // logslope designs the `*_penalties_vm` were pulled back through.
            BlockDesignCoords::PostCutover,
        )?;
        let rigid_family = make_family(
            &marginal_design,
            &logslope_design,
            initial_sigma,
            FlexActivation::OffForRigidPilot,
            BlockDesignCoords::PostCutover,
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
        // Construction-site designs: the post-cutover compiled marginal/
        // logslope designs the `*_penalties_vm` were pulled back through.
        BlockDesignCoords::PostCutover,
    )?;
    // Validate the assembled block specs at the construction boundary so any
    // design/penalty width inconsistency (e.g. a compiled-map penalty that
    // does not match its block's reduced compiled width) surfaces here as a
    // clean typed error string. Without this, the inconsistency would only be
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
        initial_sigma,
        FlexActivation::On,
        BlockDesignCoords::PostCutover,
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
    let derivative_block_cache = RefCell::new(
        None::<(
            Array1<f64>,
            Arc<Vec<Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>>>,
        )>,
    );
    let theta_matches = |left: &Array1<f64>, right: &Array1<f64>| -> bool {
        left.len() == right.len()
            && left
                .iter()
                .zip(right.iter())
                .all(|(lhs, rhs)| (*lhs - *rhs).abs() <= 1e-12 * (1.0 + lhs.abs().max(rhs.abs())))
    };
    let sigma_from_theta = |theta: &Array1<f64>| -> Option<f64> {
        initial_sigma.map(|_| theta[setup.rho_dim() + setup.log_kappa_dim()].exp())
    };
    let get_derivative_blocks = |theta: &Array1<f64>,
                                 specs: &[TermCollectionSpec],
                                 designs: &[TermCollectionDesign]|
     -> Result<
        Arc<Vec<Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>>>,
        String,
    > {
        if let Some((cached_theta, cached_blocks)) = derivative_block_cache.borrow().as_ref()
            && theta_matches(cached_theta, theta)
        {
            return Ok(Arc::clone(cached_blocks));
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
        if initial_sigma.is_some_and(|sigma| sigma > 0.0) {
            let sigma_aux = crate::custom_family::CustomFamilyBlockPsiDerivative::new(
                None,
                Array2::zeros((0, 0)),
                Array2::zeros((0, 0)),
                None,
                None,
                None,
                None,
            );
            derivative_blocks
                .last_mut()
                .ok_or_else(|| "survival marginal-slope missing derivative blocks".to_string())?
                .push(sigma_aux);
        }
        let derivative_blocks = Arc::new(derivative_blocks);
        derivative_block_cache.replace(Some((theta.clone(), Arc::clone(&derivative_blocks))));
        Ok(derivative_blocks)
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
            // Outer κ-probe / eval: `designs` are re-materialised RAW-width
            // designs from the boot specs, so the compiled `*_penalties_vm` do
            // not apply — keep the raw design-derived penalties (#788).
            let blocks = build_blocks(
                &rho,
                &designs[0],
                &designs[1],
                FlexActivation::On,
                BlockDesignCoords::RematerializedRaw,
            )?;
            let sigma = sigma_from_theta(theta);
            sigma_hint.replace(sigma);
            let family = make_family(
                &designs[0],
                &designs[1],
                sigma,
                FlexActivation::On,
                BlockDesignCoords::RematerializedRaw,
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
            // Outer κ-probe / eval: `designs` are re-materialised RAW-width
            // designs from the boot specs, so the compiled `*_penalties_vm` do
            // not apply — keep the raw design-derived penalties (#788).
            let blocks = build_blocks(
                &rho,
                &designs[0],
                &designs[1],
                FlexActivation::On,
                BlockDesignCoords::RematerializedRaw,
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
            let sigma = sigma_from_theta(theta);
            sigma_hint.replace(sigma);
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
                sigma,
                FlexActivation::On,
                BlockDesignCoords::RematerializedRaw,
            )?;
            let derivative_blocks = if matches!(effective_mode, EvalMode::ValueOnly) {
                Arc::new(vec![Vec::new(); blocks.len()])
            } else {
                get_derivative_blocks(theta, specs, designs)?
            };
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
                derivative_blocks,
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
                return Err(
                    "exact survival marginal-slope joint objective did not return an outer Hessian"
                        .to_string(),
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
            // Outer κ-probe / eval: `designs` are re-materialised RAW-width
            // designs from the boot specs, so the compiled `*_penalties_vm` do
            // not apply — keep the raw design-derived penalties (#788).
            let blocks = build_blocks(
                &rho,
                &designs[0],
                &designs[1],
                FlexActivation::On,
                BlockDesignCoords::RematerializedRaw,
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
            let sigma = sigma_from_theta(theta);
            sigma_hint.replace(sigma);
            let family = make_family(
                &designs[0],
                &designs[1],
                sigma,
                FlexActivation::On,
                BlockDesignCoords::RematerializedRaw,
            )?;
            let derivative_blocks = get_derivative_blocks(theta, specs, designs)?;
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
                derivative_blocks,
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
    let mut solved = match solved {
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
    // Recompile-after-first-PIRLS-accept refinement (math-agent review).
    //
    // The initial cutover compile used a structural identity row Hessian,
    // which catches column-level cross-block aliases but misses the
    // "pilot-curvature trap": directions that look identifiable under
    // identity H but collapse under the actual β-converged H (or, less
    // commonly, the reverse). Now that the outer/PIRLS has accepted a
    // non-trivial β, rebuild the row Hessian from the row primary state
    // at the converged β, re-run `compile_survival_parametric_designs_per_term`,
    // and compare drops_by_block. If the two compiles disagree, log a
    // warning surfacing the diff — the structural compile is still load-
    // bearing for predict-time consumers (it owns the T-lift the inner
    // Newton actually used), so we don't silently re-fit; the warning is
    // the actionable diagnostic the math agent asked for. Re-fitting once
    // with the new compile would require rebuilding all of the captured
    // post-cutover bindings (designs, make_family, build_blocks, the
    // outer solve closures), which is outside the surgical scope of this
    // hook; the diagnostic is the principled stop here.
    if n < 1_000 {
        log::debug!(
            "[smgs phase-4b recompile-after-accept] skipped for tiny fit n={n}; \
             diagnostic-only post-convergence recompile is reserved for larger fits"
        );
    } else if let Some(ref ctx) = recompile_after_accept {
        let recompile_started = std::time::Instant::now();
        // Lift compiled β → raw β when the cutover fired. Otherwise the
        // block_states already carry raw-width β.
        let n_lift = smgs_lift_v.as_ref().map(|l| l.n_blocks()).unwrap_or(0);
        let raw_core_betas: Vec<Array1<f64>> = if let Some(ref lift) = smgs_lift_v {
            let compiled_betas: Vec<Array1<f64>> = solved
                .fit
                .block_states
                .iter()
                .take(n_lift)
                .map(|s| s.beta.clone())
                .collect();
            lift.lift_block_betas(&compiled_betas)
        } else {
            solved
                .fit
                .block_states
                .iter()
                .take(3)
                .map(|state| state.beta.clone())
                .collect()
        };
        let recompile_result = (|| -> Result<(usize, usize, usize), String> {
            use crate::survival::marginal_slope::identifiability::{
                SurvivalRowHessian, compile_survival_parametric_designs_per_term,
            };
            let beta_time = raw_core_betas
                .first()
                .ok_or_else(|| "no time block β available".to_string())?;
            let beta_logslope = raw_core_betas
                .get(2)
                .ok_or_else(|| "no logslope block β available".to_string())?;
            if beta_time.len() != ctx.dq0.ncols() {
                return Err(format!(
                    "raw time β width {} != raw design width {}",
                    beta_time.len(),
                    ctx.dq0.ncols()
                ));
            }
            let n_rows = ctx.dq0.nrows();
            // Marginal η is lift-invariant. Physical logslope channels are not
            // a scalar η: evaluate them from the raw coefficient vector through
            // the canonical layout below.
            let marginal_eta = solved
                .fit
                .block_states
                .get(1)
                .map(|s| s.eta.clone())
                .ok_or_else(|| "no marginal block_state".to_string())?;
            if marginal_eta.len() != n_rows {
                return Err(format!(
                    "block_state eta length mismatch: marginal={}, n_rows={}",
                    marginal_eta.len(),
                    n_rows
                ));
            }
            let time_q0 = ctx.dq0.dot(beta_time);
            let time_q1 = ctx.dq1.dot(beta_time);
            let time_qd1 = ctx.dqd1.dot(beta_time);
            let mut q0 = Array1::<f64>::zeros(n_rows);
            let mut q1 = Array1::<f64>::zeros(n_rows);
            let mut qd1 = Array1::<f64>::zeros(n_rows);
            for i in 0..n_rows {
                q0[i] = time_q0[i] + ctx.offset_entry[i] + marginal_eta[i];
                q1[i] = time_q1[i] + ctx.offset_exit[i] + marginal_eta[i];
                qd1[i] = time_qd1[i] + ctx.derivative_offset_exit[i];
            }
            let slopes = ctx
                .logslope_layout
                .physical_values(ctx.z.ncols(), beta_logslope.view())?;
            let row_hess = SurvivalRowHessian::from_pilot_primary_state(
                &q0,
                &q1,
                &qd1,
                &slopes,
                &ctx.z,
                &ctx.covariance,
                &ctx.weights,
                &ctx.event,
                ctx.derivative_guard,
                ctx.probit_scale,
            )?;
            let compiled = compile_survival_parametric_designs_per_term(
                ctx.dq0.clone(),
                ctx.dq1.clone(),
                ctx.dqd1.clone(),
                &ctx.time_partition,
                ctx.m_dq.clone(),
                ctx.m_dqd1.clone(),
                &ctx.marginal_partition,
                ctx.logslope_layout.clone(),
                &ctx.logslope_partition,
                &row_hess,
                ctx.protect_time,
            )?;
            Ok(compiled.drops_by_block)
        })();
        match recompile_result {
            Ok(drops_post) if ctx.used_partial_logslope_reduction => {
                // #979: the cutover used the W-orthogonal PARTIAL
                // reduced-logslope reparam, whose structural drops come from
                // the effective Schur Gram (keep `wl` surviving directions).
                // The recompile above re-runs the FULL per-term compiler, which
                // by construction collapses the WHOLE logslope channel — that
                // is exactly the over-collapse the partial reparam routes
                // around, so a logslope-drop difference here is EXPECTED and
                // healthy, not a pilot-curvature trap. Confirm the full
                // compiler still over-collapses logslope at converged β (the
                // confound persists, so the partial reparam was the right
                // call) and log it at info without crying wolf.
                let confound_persists = drops_post.2 > ctx.drops_by_block_initial.2;
                log::info!(
                    "[smgs phase-4b recompile-after-accept] #979 partial reduced-logslope \
                     reparam: structural drops=(time={}, marginal={}, logslope={}); full per-term \
                     recompile at converged β drops=(time={}, marginal={}, logslope={}) — the \
                     full compiler {} over-collapses logslope, as expected for the partial \
                     reparam (no pilot-curvature trap); elapsed={:.3}s",
                    ctx.drops_by_block_initial.0,
                    ctx.drops_by_block_initial.1,
                    ctx.drops_by_block_initial.2,
                    drops_post.0,
                    drops_post.1,
                    drops_post.2,
                    if confound_persists {
                        "still"
                    } else {
                        "no longer"
                    },
                    recompile_started.elapsed().as_secs_f64(),
                );
            }
            Ok(drops_post) => {
                if drops_post == ctx.drops_by_block_initial {
                    log::debug!(
                        "[smgs phase-4b recompile-after-accept] drops match structural pass \
                         (time={}, marginal={}, logslope={}); elapsed={:.3}s",
                        drops_post.0,
                        drops_post.1,
                        drops_post.2,
                        recompile_started.elapsed().as_secs_f64(),
                    );
                } else {
                    // Re-fit ONCE would consume the new compile here. The
                    // surgical scope of this hook stops at the diagnostic;
                    // surface the diff at WARN so it cannot be missed.
                    log::warn!(
                        "[smgs phase-4b recompile-after-accept] drops_by_block differs at \
                         converged β: structural=(time={}, marginal={}, logslope={}) vs \
                         data-adaptive=(time={}, marginal={}, logslope={}); pilot-curvature \
                         trap detected — current fit reflects the structural compile. A \
                         single re-fit with the data-adaptive compile is the next step; \
                         file an issue with this log line if observed in production. \
                         elapsed={:.3}s",
                        ctx.drops_by_block_initial.0,
                        ctx.drops_by_block_initial.1,
                        ctx.drops_by_block_initial.2,
                        drops_post.0,
                        drops_post.1,
                        drops_post.2,
                        recompile_started.elapsed().as_secs_f64(),
                    );
                }
            }
            Err(reason) => {
                log::warn!(
                    "[smgs phase-4b recompile-after-accept] skipped: {reason}; elapsed={:.3}s",
                    recompile_started.elapsed().as_secs_f64(),
                );
            }
        }
    }

    let (baseline_offset_residuals, baseline_offset_curvatures) = {
        let final_family = make_family(
            &solved.designs[0],
            &solved.designs[1],
            *sigma_hint.borrow(),
            FlexActivation::On,
            BlockDesignCoords::RematerializedRaw,
        )?;
        final_family.offset_channel_geometry(&solved.fit.block_states)?
    };

    // Phase-4b V+M-exact result-time lift. When the active cutover
    // fired, the inner Newton produced θ at *compiled* width across the
    // time/marginal/logslope blocks. Predict-time consumers expect β at
    // the original raw width: `Gauge::lift_block_betas` concatenates the
    // per-block compiled θs, multiplies by the full triangular T
    // (V's on the diagonal, `−R_{a→b}` off-diagonals), and splits the
    // result at raw-block boundaries. The corresponding η is
    // numerically invariant under the lift (η = X_raw · β_raw =
    // X_raw · T · θ = X_compiled · θ) so we leave it alone. When the
    // cutover did NOT fire (smgs_lift_v is None), β is already at raw
    // width and the lift is a no-op. Flex blocks (score_warp_dev,
    // link_dev) at indices ≥ 3 are not part of the parametric T; the
    // gauge is extended with identity blocks over their widths so the
    // joint covariance lift below sees them pass through unchanged.
    if let Some(ref lift) = smgs_lift_v {
        let n_lift = lift.n_blocks();
        // Flex-block widths BEFORE the β lift (identical after — flex
        // blocks are never compiled), used to extend the gauge so joint
        // (compiled+flex)-width matrices lift in one sandwich.
        let flex_widths: Vec<usize> = solved
            .fit
            .blocks
            .iter()
            .skip(n_lift)
            .map(|b| b.beta.len())
            .collect();
        let compiled_betas: Vec<Array1<f64>> = solved
            .fit
            .block_states
            .iter()
            .take(n_lift)
            .map(|s| s.beta.clone())
            .collect();
        let lifted = lift.lift_block_betas(&compiled_betas);
        for ((state, block), beta) in solved
            .fit
            .block_states
            .iter_mut()
            .take(n_lift)
            .zip(solved.fit.blocks.iter_mut().take(n_lift))
            .zip(lifted.into_iter())
        {
            state.beta = beta.clone();
            block.beta = beta;
        }
        let mut off = 0usize;
        let total: usize = solved.fit.blocks.iter().map(|b| b.beta.len()).sum();
        let mut flat = Array1::<f64>::zeros(total);
        for block in &solved.fit.blocks {
            let p = block.beta.len();
            flat.slice_mut(ndarray::s![off..off + p])
                .assign(&block.beta);
            off += p;
        }
        solved.fit.beta = flat;

        // Push coefficient covariance into the raw reporting frame and retain
        // precision in the active frame with the SAME affine gauge used for
        // beta (#741, #933, #2301). A rectangular T has no raw-space precision:
        // T H T' would be a singular covariance-style congruence, not the
        // optimizer's quadratic form. Saved ALO instead uses J_raw T against
        // the unchanged active H. The joint matrices span compiled parametric
        // blocks followed by raw flex blocks, so the gauge is extended with
        // identity over the flex widths.
        let joint_gauge = lift.extend_with_identity(&flex_widths);
        let lift_joint_covariance = |name: &str,
                                     matrix: Array2<f64>|
         -> Result<Array2<f64>, String> {
            let active = joint_gauge.reduced_total();
            if matrix.dim() != (active, active) {
                return Err(format!(
                    "survival marginal-slope {name} is {}x{}; exact result gauge requires {active}x{active} active coordinates before its {}-coefficient raw lift",
                    matrix.nrows(),
                    matrix.ncols(),
                    joint_gauge.raw_total(),
                ));
            }
            Ok(joint_gauge.lift_covariance(&matrix))
        };
        solved.fit.covariance_conditional = solved
            .fit
            .covariance_conditional
            .take()
            .map(|covariance| lift_joint_covariance("covariance_conditional", covariance))
            .transpose()?;
        solved.fit.covariance_corrected = solved
            .fit
            .covariance_corrected
            .take()
            .map(|covariance| lift_joint_covariance("covariance_corrected", covariance))
            .transpose()?;
        if let Some(mut geometry) = solved.fit.geometry.take() {
            geometry.coefficient_gauge = geometry
                .coefficient_gauge
                .left_compose(&joint_gauge)
                .map_err(|reason| {
                    format!(
                        "survival marginal-slope active geometry cannot compose with its exact raw result gauge: {reason}"
                    )
                })?;
            solved.fit.geometry = Some(geometry);
        }
    }

    let mut resolved_specs = solved.resolved_specs;
    let designs = solved.designs;
    Ok(SurvivalMarginalSlopeFitResult {
        fit: solved.fit,
        marginalspec_resolved: resolved_specs.remove(0),
        logslopespec_resolved: resolved_specs.remove(0),
        marginal_design: designs[0].clone(),
        logslope_design: designs[1].clone(),
        gaussian_frailty_sd: *sigma_hint.borrow(),
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
