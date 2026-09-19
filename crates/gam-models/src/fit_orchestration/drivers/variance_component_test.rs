// The per-term variance-component test — the driver that turns a fitted
// standard GAM into "does each `group()`/`re()` block, and each smooth whose
// penalties cover every direction, carry an effect?".
//
// `include!`d into `drivers/mod.rs` like the other self-contained inference
// subsystems, so it shares the driver's flat namespace and import surface.
//
// The statistic and its exact reference law live in
// `gam_terms::inference::variance_component_test`. This file only reads the
// fit's retained IRLS row state, the realized coefficient layout and the
// realized penalties and hands them over; every tested term gets a record, and
// a term the test cannot score carries the typed reason instead of a p-value.

/// The variance-component test of every random-effect block (or, for an
/// unpenalized factor block, the fixed-effect test) and of every smooth whose
/// active penalties jointly penalize every direction
/// ([`gam_terms::smooth::TermCollectionDesign::smooth_variance_component_penalties`]).
///
/// Never fails: a fit without the row state the score needs yields one
/// `NoIrlsRowState` record per term, so the summary always has an answer for
/// each row it routes to this test.
pub fn variance_component_test_records(
    design: &gam_terms::smooth::TermCollectionDesign,
    fit: &UnifiedFitResult,
) -> Vec<gam_terms::inference::variance_component_test::VarianceComponentTestRecord> {
    use gam_terms::inference::variance_component_test::{
        VarianceComponentTermRequest, VarianceComponentTestBasis, VarianceComponentTestInput, VarianceComponentTestOutcome, VarianceComponentTestRecord,
        VarianceComponentTestScale, VarianceComponentTestUnavailable,
    };

    let tested = variance_component_test_terms(design);
    if tested.is_empty() {
        return Vec::new();
    }
    let records = |outcomes: Vec<VarianceComponentTestOutcome>| {
        tested
            .iter()
            .zip(outcomes)
            .map(|((name, request), outcome)| VarianceComponentTestRecord {
                term: name.clone(),
                coefficient_range: request.range.clone(),
                outcome,
            })
            .collect::<Vec<_>>()
    };
    let unavailable = |reason: VarianceComponentTestUnavailable| {
        records(
            tested
                .iter()
                .map(|_| VarianceComponentTestOutcome::Unavailable { reason })
                .collect(),
        )
    };

    let n_rows = design.design.nrows();
    let Some(rows_state) = score_row_state(fit, n_rows) else {
        return unavailable(VarianceComponentTestUnavailable::NoIrlsRowState);
    };
    if fit.beta.len() != design.design.ncols() {
        return unavailable(VarianceComponentTestUnavailable::DesignUnavailable);
    }
    // Same scale contract as the basis-adequacy score test: a profiled
    // dispersion is estimated, otherwise the score's variance is scaled by the
    // multiplier the fit publishes on its coefficient covariance.
    let scale = if fit.likelihood_scale.wald_scale_is_estimated() {
        VarianceComponentTestScale::Estimated
    } else {
        VarianceComponentTestScale::Known {
            dispersion: fit
                .coefficient_covariance_scale()
                .ok()
                .filter(|value| value.is_finite() && *value > 0.0)
                .unwrap_or(1.0),
        }
    };
    let basis = match VarianceComponentTestBasis::new(VarianceComponentTestInput {
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
    let requests: Vec<VarianceComponentTermRequest> =
        tested.iter().map(|(_, request)| request.clone()).collect();
    records(
        basis
            .test_terms(&requests)
            .into_iter()
            .map(VarianceComponentTestOutcome::from)
            .collect(),
    )
}

/// The terms the variance-component test scores, named, with their GLOBAL
/// coefficient ranges and null.
///
/// A random-effect block is tested against the penalties the design realized
/// on exactly its range (none: an unpenalized factor block, a fixed effect).
/// A smooth is tested when
/// [`gam_terms::smooth::TermCollectionDesign::smooth_variance_component_penalties`]
/// admits it; every other smooth keeps the Wald test.
fn variance_component_test_terms(
    design: &gam_terms::smooth::TermCollectionDesign,
) -> Vec<(
    String,
    gam_terms::inference::variance_component_test::VarianceComponentTermRequest,
)> {
    use gam_terms::inference::variance_component_test::{
        TestedBlock, VarianceComponentTermRequest,
    };

    let mut terms = Vec::new();
    for (name, range) in &design.random_effect_ranges {
        let penalties: Vec<Array2<f64>> = design
            .penalties
            .iter()
            .filter(|penalty| penalty.col_range == *range)
            .map(|penalty| penalty.local.clone())
            .collect();
        let block = if penalties.is_empty() {
            TestedBlock::Unpenalized
        } else {
            TestedBlock::Penalized { penalties }
        };
        terms.push((
            name.clone(),
            VarianceComponentTermRequest {
                range: range.clone(),
                block,
            },
        ));
    }
    for term in &design.smooth.terms {
        if let Some(penalties) = design.smooth_variance_component_penalties(term) {
            terms.push((
                term.name.clone(),
                VarianceComponentTermRequest {
                    range: design.smooth_global_range(term),
                    block: TestedBlock::Penalized { penalties },
                },
            ));
        }
    }
    terms
}
