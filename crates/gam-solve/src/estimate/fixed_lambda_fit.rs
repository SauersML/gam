//! The nested model of a likelihood-ratio test, fitted at the full model's
//! smoothing parameters.
//!
//! A per-term LR test compares the fitted model against the same model with
//! one coefficient block constrained to zero. Both sides of that comparison
//! have to be the SAME penalized likelihood with one constraint added, or the
//! statistic measures something other than the constraint. The smoothing
//! parameters are part of that likelihood: re-selecting them for the reduced
//! model moves every surviving term's `λ` along with the dropped block, and
//! when the tested term sits at its penalty null space (`λ̂` on the ceiling) the
//! two REML optima differ only by the outer search's own tolerance. That
//! tolerance then IS the statistic — observed as `W ~ 1e-6 … 1e-4` against a
//! null law whose mean is `~1e-8`, published as `p < 1e-237`.
//!
//! [`fit_nested_at_fitted_log_lambdas`] therefore fits the reduced model at the
//! full fit's own `ρ̂` (each surviving penalty keeps its `λ̂`, the dropped
//! block's penalties leave with it), with no outer search. The inner solve is
//! the same P-IRLS the full fit's final accept-solve ran, on the same
//! likelihood, link and Firth setting, so the only difference between the two
//! optima is the constraint `β_block = 0`.
//!
//! Nuisance parameters follow the full fit's final solve: a dispersion the inner
//! solve refines at the converged `η` (Gamma shape, Beta precision) is refined
//! for the reduced model too, and one that the outer search owns (negative
//! binomial `θ`, Student-t `σ, ν`) stays at its fitted value. The profiled
//! Gaussian scale is re-profiled from the reduced fit's own residual degrees of
//! freedom, which the caller receives so the estimated-scale reference can
//! absorb the deterministic part of `W`. A flexible link whose shape the outer
//! search estimated is not information-orthogonal to `β`, so holding it fixed
//! would not give the profile LR; that case is declined by name.

use super::*;
use std::ops::Range;

/// The full fit and its realized problem, as the full fit's solver saw them.
pub struct NestedFixedLambdaInputs<'a> {
    /// The full model's design (original, unconditioned columns).
    pub design: &'a DesignMatrix,
    pub y: ArrayView1<'a, f64>,
    pub weights: ArrayView1<'a, f64>,
    /// The offset the full fit was solved with (user offset plus the design's
    /// fixed affine channel).
    pub offset: ArrayView1<'a, f64>,
    /// The full model's penalties, in the order the full fit received them.
    pub penalties: &'a [BlockwisePenalty],
    /// One null-space dimension per entry of `penalties`.
    pub nullspace_dims: &'a [usize],
    /// The full model's linear inequality constraints, in original columns.
    pub linear_constraints: Option<&'a gam_problem::LinearInequalityConstraints>,
    /// The converged full fit.
    pub fit: &'a UnifiedFitResult,
    /// The tolerance the full fit was solved to.
    pub tol: f64,
    /// Whether the outer search estimated a flexible link's shape (SAS,
    /// Beta-logistic, mixture weights) rather than holding it fixed.
    pub link_shape_estimated: bool,
}

/// The reduced model's optimum at the full fit's `ρ̂`.
#[derive(Clone, Debug)]
pub struct NestedFixedLambdaFit {
    /// Fully normalized log-likelihood at the reduced optimum, under the same
    /// reporting convention as the full fit's `log_likelihood`.
    pub log_likelihood: f64,
    /// The reduced model's linear predictor, offset included.
    pub eta: Array1<f64>,
    /// Profiled Gaussian only: the residual degrees of freedom `ν₀ = n₊ − edf₀`
    /// whose `σ̂₀² = D₀/ν₀` the log-likelihood was evaluated at.
    pub profiled_residual_df: Option<f64>,
    /// `∂ℓ_i/∂η_i` at the reduced optimum, in the units of the full fit's
    /// unscaled Hessian `H = XᵀWX + S`: under the fitted dispersion for every
    /// family whose working weight carries it, and at unit dispersion for a
    /// profiled Gaussian, whose `H` is written at `σ² = 1`.
    ///
    /// `X_jᵀ` of it is the score of the dropped block at the constrained
    /// optimum, where every kept block's penalized score is zero.
    pub eta_score: Array1<f64>,
}

/// Outcome of the reduced fit. Only [`Self::Converged`] carries a likelihood:
/// a solve that did not certify its optimum yields no fit.
#[derive(Clone, Debug)]
pub enum NestedFixedLambdaOutcome {
    Converged(NestedFixedLambdaFit),
    /// The inner solve did not reach a certified optimum.
    NotConverged(String),
    /// The reduced model is not the full model with one block constrained to
    /// zero at fixed `ρ̂` (a penalty straddles the block, a constraint becomes
    /// infeasible, the link shape was estimated, …).
    Unsupported(String),
}

/// Fit the full model with coefficient columns `dropped` removed, at the full
/// fit's smoothing parameters.
///
/// `Err` is reserved for inputs that break this function's own invariants
/// (mismatched lengths, malformed penalties); every property of the data or the
/// model that prevents the reduced fit is a typed [`NestedFixedLambdaOutcome`].
pub fn fit_nested_at_fitted_log_lambdas(
    inputs: &NestedFixedLambdaInputs<'_>,
    dropped: Range<usize>,
) -> Result<NestedFixedLambdaOutcome, EstimationError> {
    const CONTEXT: &str = "nested fixed-lambda fit";
    let fit = inputs.fit;
    let p = inputs.design.ncols();
    let n = inputs.design.nrows();
    if let Some(message) = row_mismatch_message(
        inputs.y.len(),
        inputs.weights.len(),
        n,
        inputs.offset.len(),
    ) {
        crate::bail_invalid_estim!("{CONTEXT}: {message}");
    }
    if dropped.start >= dropped.end || dropped.end > p {
        crate::bail_invalid_estim!(
            "{CONTEXT}: dropped columns {:?} are not a non-empty range within {p} columns",
            dropped
        );
    }
    if inputs.penalties.len() != inputs.nullspace_dims.len() {
        crate::bail_invalid_estim!(
            "{CONTEXT}: {} penalties but {} null-space dimensions",
            inputs.penalties.len(),
            inputs.nullspace_dims.len()
        );
    }
    let q = dropped.len();
    let p_null = p - q;

    // The full fit's ρ̂ indexes the penalties that survived canonicalization, in
    // order; pair each surviving penalty with its coordinate, then keep the
    // ones outside the dropped block.
    let mut kept_specs = Vec::<PenaltySpec>::new();
    let mut kept_nullspace_dims = Vec::<usize>::new();
    let mut kept_rho = Vec::<f64>::new();
    let mut coordinate = 0usize;
    for (index, penalty) in inputs.penalties.iter().enumerate() {
        let spec = PenaltySpec::from_blockwise_ref(penalty);
        if gam_terms::construction::canonicalize_penalty_spec(&spec, p, index, CONTEXT)?.is_none() {
            continue;
        }
        let Some(&rho) = fit.log_lambdas.get(coordinate) else {
            return Ok(NestedFixedLambdaOutcome::Unsupported(format!(
                "the full fit carries {} smoothing coordinates for more active penalties",
                fit.log_lambdas.len()
            )));
        };
        coordinate += 1;
        let range = &penalty.col_range;
        let col_range = if range.start >= dropped.start && range.end <= dropped.end {
            continue;
        } else if range.start >= dropped.end {
            (range.start - q)..(range.end - q)
        } else if range.end <= dropped.start {
            range.clone()
        } else {
            return Ok(NestedFixedLambdaOutcome::Unsupported(format!(
                "penalty {index} on columns {range:?} straddles the tested block {dropped:?}"
            )));
        };
        kept_specs.push(PenaltySpec::Block {
            local: penalty.local.clone(),
            col_range,
            structure_hint: penalty.structure_hint.clone(),
            op: penalty.op.clone(),
        });
        kept_nullspace_dims.push(inputs.nullspace_dims[index]);
        kept_rho.push(rho);
    }
    if coordinate != fit.log_lambdas.len() {
        return Ok(NestedFixedLambdaOutcome::Unsupported(format!(
            "the full fit carries {} smoothing coordinates for {coordinate} active penalties",
            fit.log_lambdas.len()
        )));
    }
    let (canonical, _) = gam_terms::construction::canonicalize_penalty_specs(
        &kept_specs,
        &kept_nullspace_dims,
        p_null,
        CONTEXT,
    )?;
    if canonical.len() != kept_rho.len() {
        return Ok(NestedFixedLambdaOutcome::Unsupported(format!(
            "{} of the {} surviving penalties stay active on the reduced design",
            canonical.len(),
            kept_rho.len()
        )));
    }
    let rho = Array1::from_vec(kept_rho);

    let kept_columns: Vec<usize> = (0..dropped.start).chain(dropped.end..p).collect();
    let x_reduced = inputs
        .design
        .select_columns(&kept_columns)
        .map_err(EstimationError::InvalidInput)?;

    // A constraint row that loses every nonzero coefficient becomes `0 ≥ b`:
    // vacuous when `b ≤ 0`, and a model the constraint set excludes when `b > 0`.
    let constraints = match inputs.linear_constraints {
        None => None,
        Some(full) => {
            let mut rows = Vec::<usize>::new();
            for (row, a_row) in full.a.outer_iter().enumerate() {
                let active = kept_columns.iter().any(|&column| a_row[column] != 0.0);
                if active {
                    rows.push(row);
                } else if full.b[row] > 0.0 {
                    return Ok(NestedFixedLambdaOutcome::Unsupported(format!(
                        "constraint row {row} involves only the tested block and is \
                         infeasible without it"
                    )));
                }
            }
            if rows.is_empty() {
                None
            } else {
                let a = full
                    .a
                    .select(Axis(0), &rows)
                    .select(Axis(1), &kept_columns);
                let b = full.b.select(Axis(0), &rows);
                Some(
                    gam_problem::LinearInequalityConstraints::new(a, b)
                        .map_err(EstimationError::InvalidInput)?,
                )
            }
        }
    };

    // The same column conditioning the full fit's solver applies, inferred from
    // the reduced problem so the unpenalized columns it rescales are the ones
    // the reduced design still has.
    let conditioning = ParametricColumnConditioning::infer_from_penalty_specs(&x_reduced, &kept_specs);
    let x_fit = conditioning.apply_to_design(&x_reduced);
    let fit_constraints = conditioning.transform_linear_constraints_to_internal(constraints);

    let Some(family) = fit.likelihood_family.clone() else {
        return Ok(NestedFixedLambdaOutcome::Unsupported(
            "the full fit records no likelihood family".to_string(),
        ));
    };
    let likelihood = GlmLikelihoodSpec::try_new(family, fit.likelihood_scale).map_err(|error| {
        EstimationError::InvalidInput(format!("{CONTEXT}: full fit likelihood: {error:?}"))
    })?;
    let link_kind = match &fit.fitted_link {
        FittedLinkState::Standard(_) | FittedLinkState::LatentCLogLog { .. } => {
            likelihood.spec.link.clone()
        }
        FittedLinkState::Sas { .. }
        | FittedLinkState::BetaLogistic { .. }
        | FittedLinkState::Mixture { .. }
            if inputs.link_shape_estimated =>
        {
            return Ok(NestedFixedLambdaOutcome::Unsupported(
                "the link shape was estimated jointly with the mean; holding it at the \
                 full fit's value is not the profile likelihood ratio"
                    .to_string(),
            ));
        }
        FittedLinkState::Sas { state, .. } => InverseLink::Sas(state.clone()),
        FittedLinkState::BetaLogistic { state, .. } => InverseLink::BetaLogistic(state.clone()),
        FittedLinkState::Mixture { state, .. } => InverseLink::Mixture(state.clone()),
    };
    let mut config =
        RemlConfig::external(likelihood, inputs.tol, fit.artifacts.firth_bias_reduction)
            .as_pirls_config();
    config.link_kind = link_kind;

    let solved = pirls::fit_model_for_fixed_rho_with_adaptive_kkt(
        LogSmoothingParamsView::new(rho.view())?,
        pirls::PirlsProblem {
            x: &x_fit,
            offset: inputs.offset,
            y: inputs.y,
            priorweights: inputs.weights,
            covariate_se: None,
            gaussian_fixed_cache: None,
            glm_first_step_gram: None,
        },
        pirls::PenaltyConfig {
            canonical_penalties: &canonical,
            reparam_invariant: None,
            p: p_null,
            coefficient_lower_bounds: None,
            linear_constraints_original: fit_constraints.as_ref(),
        },
        &config,
        None,
        None,
        // As in the full fit's final solve: refine the dispersion the inner
        // solve owns at the converged η.
        true,
        None,
    );
    let result = match solved {
        Ok((result, _)) => result,
        Err(error) => return Ok(NestedFixedLambdaOutcome::NotConverged(error.to_string())),
    };
    if !matches!(result.status, pirls::PirlsStatus::Converged) {
        return Ok(NestedFixedLambdaOutcome::NotConverged(format!(
            "inner solve ended {:?}",
            result.status
        )));
    }

    // Report under the full fit's convention: a profiled Gaussian scale is
    // evaluated at its residual-df estimate `σ̂₀² = D₀/ν₀`.
    let mut reporting = result.likelihood.clone();
    let mut profiled_residual_df = None;
    if matches!(reporting.spec.response, ResponseFamily::Gaussian)
        && matches!(reporting.scale, LikelihoodScaleMetadata::ProfiledGaussian)
    {
        let observations = inputs.weights.iter().filter(|weight| **weight > 0.0).count() as f64;
        let residual_df = observations - result.edf;
        let deviance = result.deviance;
        if !(residual_df > 0.0 && residual_df <= observations && deviance > 0.0) {
            return Ok(NestedFixedLambdaOutcome::Unsupported(format!(
                "the reduced Gaussian fit has no positive residual scale \
                 (deviance {deviance}, residual df {residual_df})"
            )));
        }
        reporting.scale = LikelihoodScaleMetadata::FixedDispersion {
            phi: deviance / residual_df,
        };
        profiled_residual_df = Some(residual_df);
    }
    let log_likelihood = pirls::evaluate_full_log_likelihood_from_eta(
        inputs.y,
        result.final_eta.view(),
        &reporting,
        inputs.weights,
    )?
    .total();
    let eta = result.final_eta.to_owned();
    let mut score_likelihood = reporting;
    if profiled_residual_df.is_some() {
        score_likelihood.scale = LikelihoodScaleMetadata::FixedDispersion { phi: 1.0 };
    }
    let mut eta_score = Array1::<f64>::zeros(eta.len());
    pirls::eta_log_likelihood_value_and_score_into(
        inputs.y,
        &eta,
        &score_likelihood,
        &config.link_kind,
        inputs.weights,
        &mut eta_score,
    )?;
    Ok(NestedFixedLambdaOutcome::Converged(NestedFixedLambdaFit {
        log_likelihood,
        eta,
        profiled_residual_df,
        eta_score,
    }))
}
