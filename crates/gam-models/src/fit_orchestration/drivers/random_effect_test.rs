// The per-term variance-component test — the driver that turns a fitted
// standard GAM into "does each `group()` block carry a between-group effect?"
// and "does each ridged linear term carry any effect?" (#3573).
//
// `include!`d into `drivers/mod.rs` like the other self-contained inference
// subsystems, so it shares the driver's flat namespace and import surface.
//
// The statistic and its exact reference law live in
// `gam_terms::inference::random_effect_test`. This file only reads the fit's
// retained IRLS row state and the realized coefficient layout and hands them
// over; every term gets a record, and a term the test cannot score carries the
// typed reason instead of a p-value.

/// The variance-component score tests of a fitted standard GAM: one record
/// per random-effect block and one per linear term carrying the null-recovery
/// ridge (`TermCollectionDesign::ridged_linear_ranges`, #3573).
pub struct VarianceComponentTestRecords {
    pub random_effect: Vec<gam_terms::inference::random_effect_test::RandomEffectTestRecord>,
    pub linear_term: Vec<gam_terms::inference::random_effect_test::RandomEffectTestRecord>,
}

/// The variance-component test of every random-effect block and every ridged
/// linear term of a fitted standard GAM, scored against one shared basis.
///
/// Never fails: a fit without the row state the score needs yields one
/// `NoIrlsRowState` record per term, so the summary always has an answer for
/// each such row.
pub fn variance_component_test_records(
    design: &gam_terms::smooth::TermCollectionDesign,
    fit: &UnifiedFitResult,
) -> VarianceComponentTestRecords {
    use gam_terms::inference::random_effect_test::{
        RandomEffectTermRequest, RandomEffectTestBasis, RandomEffectTestInput,
        RandomEffectTestOutcome, RandomEffectTestRecord, RandomEffectTestScale,
        RandomEffectTestUnavailable,
    };

    let random_effect_count = design.random_effect_ranges.len();
    let ranges: Vec<(String, std::ops::Range<usize>)> = design
        .random_effect_ranges
        .iter()
        .cloned()
        .chain(design.ridged_linear_ranges())
        .collect();
    let split = |outcomes: Vec<RandomEffectTestOutcome>| {
        let mut records: Vec<RandomEffectTestRecord> = ranges
            .iter()
            .zip(outcomes)
            .map(|((name, range), outcome)| RandomEffectTestRecord {
                term: name.clone(),
                coefficient_range: range.clone(),
                outcome,
            })
            .collect();
        let linear_term = records.split_off(random_effect_count);
        VarianceComponentTestRecords {
            random_effect: records,
            linear_term,
        }
    };
    if ranges.is_empty() {
        return split(Vec::new());
    }
    let unavailable = |reason: RandomEffectTestUnavailable| {
        split(
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
    // multiplier the fit publishes on its coefficient covariance. A known scale
    // the fit cannot publish is a typed reason, never a unit dispersion: a
    // substituted 1 would calibrate every p-value against the wrong variance.
    let scale = if fit.likelihood_scale.wald_scale_is_estimated() {
        RandomEffectTestScale::Estimated
    } else {
        match fit.coefficient_covariance_scale() {
            Ok(dispersion) if dispersion.is_finite() && dispersion > 0.0 => {
                RandomEffectTestScale::Known { dispersion }
            }
            _ => return unavailable(RandomEffectTestUnavailable::KnownScaleUnavailable),
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
        .map(|(_, range)| RandomEffectTermRequest {
            range: range.clone(),
        })
        .collect();
    split(
        basis
            .test_terms(&requests)
            .into_iter()
            .map(RandomEffectTestOutcome::from)
            .collect(),
    )
}
