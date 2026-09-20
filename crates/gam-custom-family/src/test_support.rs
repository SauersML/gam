//! Test-only helpers: the internal outer objective evaluator from one seed, as a
//! dense (objective, gradient, Hessian, warm-start) tuple for finite-difference
//! checks. Kept as a sibling `#[cfg(test)] mod test_support` so the test module
//! reaches it via `super::test_support::...` exactly as before.

use super::*;
use ndarray::{Array1, Array2};

pub(crate) fn outerobjectivegradienthessian<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
    penalty_counts: &[usize],
    rho: &Array1<f64>,
    warm_start: Option<&ConstrainedWarmStart>,
    eval_mode: EvalMode,
) -> Result<(f64, Array1<f64>, Option<Array2<f64>>, ConstrainedWarmStart), String> {
    let result = outerobjectivegradienthessian_internal(
        family,
        specs,
        options,
        penalty_counts,
        rho,
        warm_start,
        gam_problem::RhoPrior::Flat,
        eval_mode,
    )
    .map_err(|error| error.to_string())?;
    Ok((
        result.objective,
        result.gradient,
        result
            .outer_hessian
            .materialize_dense()
            .map_err(|error| error.to_string())?,
        result.warm_start,
    ))
}

/// One outer evaluation solved directly from one seed and priced in `eval_mode`, with no
/// continuation and no other start. Production evaluations publish the mode the gam#3173 rule
/// selects among their starts ([`evaluate_on_branch`]); tests that price the criterion at a mode
/// they chose use this.
pub(crate) fn outerobjectivegradienthessian_internal<
    F: CustomFamily + Clone + Send + Sync + 'static,
>(
    family: &F,
    specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
    penalty_counts: &[usize],
    rho: &Array1<f64>,
    warm_start: Option<&ConstrainedWarmStart>,
    rho_prior: gam_problem::RhoPrior,
    eval_mode: EvalMode,
) -> Result<OuterObjectiveEvalResult, CustomFamilyError> {
    let hyper_layout = CustomFamilyHyperLayout::new(
        vec![Vec::<CustomFamilyBlockPsiDerivative>::new(); specs.len()],
        Vec::new(),
        Array1::zeros(0),
    )?;
    evaluate_custom_family_hyper_internal(
        family,
        specs,
        options,
        penalty_counts,
        rho,
        &hyper_layout,
        warm_start,
        rho_prior,
        eval_mode,
    )
}

/// [`outerobjectivegradienthessian_internal`] at a labeled `rho`: the joint penalty bundle built
/// from the outer ρ, and the evaluation pulled back to the labeled coordinates.
pub(crate) fn outerobjectivegradienthessian_labeled<
    F: CustomFamily + Clone + Send + Sync + 'static,
>(
    family: &F,
    specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
    layout: &PenaltyLabelLayout,
    rho: &Array1<f64>,
    warm_start: Option<&ConstrainedWarmStart>,
    rho_prior: &gam_problem::RhoPrior,
    eval_mode: EvalMode,
) -> Result<OuterObjectiveEvalResult, CustomFamilyError> {
    let physical_rho = expand_labeled_log_lambdas(rho, layout)?;
    let physical_warm_start = physical_warm_start_for_labeled(warm_start, &physical_rho, layout);
    // gam#1587: build the per-eval joint penalty bundle from the current outer ρ
    // (each joint spec's λ pulled from its tied outer coordinate) and attach it
    // to the inner-solve options so BOTH the inner β̂ AND the outer evaluator
    // (penalty coords / logdet / operator) see the full-width centered penalty.
    // No joint specs ⇒ `options` is passed through untouched (byte-identical).
    let labeled_options = labeled_options_for_rho(options, specs, layout, rho)?;
    let options = labeled_options.as_ref();
    let base = outerobjectivegradienthessian_internal(
        family,
        specs,
        options,
        &layout.penalty_counts,
        &physical_rho,
        physical_warm_start.as_ref().or(warm_start),
        gam_problem::RhoPrior::Flat,
        eval_mode,
    )?;
    pullback_labeled_outer_eval(base, rho, layout, rho_prior, eval_mode)
        .map_err(CustomFamilyError::from)
}
