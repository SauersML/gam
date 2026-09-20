// The per-term variance-component test — the driver that turns a fitted
// standard GAM into "does each `group()` block carry a between-group effect?"
// and "does each ridged slope carry an effect?".
//
// `include!`d into `drivers/mod.rs` like the other self-contained inference
// subsystems, so it shares the driver's flat namespace and import surface.
//
// The statistic and its exact reference law live in
// `gam_terms::inference::random_effect_test`. This file only reads the fit's
// retained IRLS row state and the realized coefficient layout and hands them
// over; every term gets a record, and a term the test cannot score carries the
// typed reason instead of a p-value.

/// The variance-component test of every random-effect block and every
/// `LinearTermRidge`-penalized linear term of a fitted standard GAM.
///
/// A ridged slope `β_j` with penalty `λ_j m_j β_j²` is the variance component
/// `β_j ~ N(0, φ/(λ_j m_j))`, and "no effect" is `λ_j = ∞`, on the boundary:
/// the same question a `group()` block asks, with a one-column design. The
/// fit's `β̂_j` was shrunk by a `λ_j` REML chose from the same data, so
/// `β̂_j/se` has no valid Wald reference (it piles its null p-values up at one);
/// the score statistic reads no `β̂_j`. For a single column it is the Rao
/// score statistic of `β_j = 0` with every other column projected out, and for
/// a Gaussian model with an estimated scale it is exactly the partial `t²` of
/// the unpenalized slope. The summary's parametric rows read these records.
///
/// Never fails: a fit without the row state the score needs yields one
/// `NoIrlsRowState` record per term, so the summary always has an answer for
/// each tested row.
pub fn random_effect_test_records(
    design: &gam_terms::smooth::TermCollectionDesign,
    fit: &UnifiedFitResult,
) -> Vec<gam_terms::inference::random_effect_test::RandomEffectTestRecord> {
    use gam_terms::inference::random_effect_test::{
        RandomEffectTermRequest, RandomEffectTestBasis, RandomEffectTestInput,
        RandomEffectTestOutcome, RandomEffectTestRecord, RandomEffectTestScale,
        RandomEffectTestUnavailable,
    };

    let ranges: Vec<(String, std::ops::Range<usize>)> = design
        .ridged_linear_ranges()
        .into_iter()
        .chain(design.random_effect_ranges.iter().cloned())
        .collect();
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
        .map(|(_, range)| RandomEffectTermRequest {
            range: range.clone(),
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
