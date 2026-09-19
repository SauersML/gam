// The per-term random-effect test — the driver that turns a fitted standard GAM
// into "does each `group()`/`re()` block carry a between-group effect?".
//
// `include!`d into `drivers/mod.rs` like the other self-contained inference
// subsystems, so it shares the driver's flat namespace and import surface.
//
// The statistic and its exact reference law live in
// `gam_terms::inference::random_effect_test`. This file only reads the fit's
// retained IRLS row state and the realized coefficient layout and hands them
// over; every term gets a record, and a term the test cannot score carries the
// typed reason instead of a p-value.

/// The variance-component (or, for an unpenalized factor block, fixed-effect)
/// test of every random-effect block of a fitted standard GAM.
///
/// Never fails: a fit without the row state the score needs yields one
/// `NoIrlsRowState` record per block, so the summary always has an answer for
/// each random-effect row.
pub fn random_effect_test_records(
    design: &gam_terms::smooth::TermCollectionDesign,
    spec: &gam_terms::smooth::TermCollectionSpec,
    fit: &UnifiedFitResult,
) -> Vec<gam_terms::inference::random_effect_test::RandomEffectTestRecord> {
    use gam_terms::inference::random_effect_test::{
        RandomEffectTermRequest, RandomEffectTestBasis, RandomEffectTestInput,
        RandomEffectTestOutcome, RandomEffectTestRecord, RandomEffectTestScale,
        RandomEffectTestUnavailable,
    };

    let ranges = &design.random_effect_ranges;
    if ranges.is_empty() {
        return Vec::new();
    }
    let records = |outcomes: Vec<RandomEffectTestOutcome>| -> Vec<RandomEffectTestRecord> {
        ranges
            .iter()
            .zip(outcomes)
            .map(|((name, range), outcome)| RandomEffectTestRecord {
                term: name.clone(),
                coefficient_range: range.clone(),
                outcome,
            })
            .collect()
    };
    let unavailable = |reason: RandomEffectTestUnavailable| {
        records(
            ranges
                .iter()
                .map(|_| RandomEffectTestOutcome::Unavailable { reason })
                .collect(),
        )
    };

    let n_rows = design.design.nrows();
    let Some(rows_state) = score_row_state(fit, n_rows) else {
        return unavailable(RandomEffectTestUnavailable::NoIrlsRowState);
    };
    if fit.beta.len() != design.design.ncols() {
        return unavailable(RandomEffectTestUnavailable::DesignUnavailable);
    }
    // Same scale contract as the basis-adequacy score test: a profiled
    // dispersion is estimated, otherwise the score's variance is scaled by the
    // multiplier the fit publishes on its coefficient covariance.
    let scale = if fit.likelihood_scale.wald_scale_is_estimated() {
        RandomEffectTestScale::Estimated
    } else {
        RandomEffectTestScale::Known {
            dispersion: fit
                .coefficient_covariance_scale()
                .ok()
                .filter(|value| value.is_finite() && *value > 0.0)
                .unwrap_or(1.0),
        }
    };
    let basis = match RandomEffectTestBasis::new(RandomEffectTestInput {
        design: &design.design,
        beta: fit.beta.view(),
        hessian_weights: rows_state.hessian_weights.view(),
        score_weights: rows_state.score_weights.view(),
        score: rows_state.score.view(),
        scale,
    }) {
        Ok(basis) => basis,
        Err(reason) => return unavailable(reason),
    };
    let requests: Vec<RandomEffectTermRequest> = ranges
        .iter()
        .map(|(name, range)| RandomEffectTermRequest {
            range: range.clone(),
            penalized: spec
                .random_effect_terms
                .iter()
                .find(|term| term.name == *name)
                .is_none_or(|term| term.penalized),
        })
        .collect();
    records(
        basis
            .test_terms(&requests)
            .into_iter()
            .map(RandomEffectTestOutcome::from)
            .collect(),
    )
}
