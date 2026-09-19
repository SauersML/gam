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

/// The variance-component test of every random-effect block and of every smooth whose
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
        VarianceComponentTestScale, VarianceComponentTestUnavailable,
    };

    let rows = score_row_state(fit, design.design.nrows())
        .ok_or(VarianceComponentTestUnavailable::NoIrlsRowState);
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
    variance_component_test_records_from_rows(design, fit.beta.view(), rows, scale)
}

/// The fitted state of a Gaussian location-scale mean block,
/// `y ~ N(μ, σ²)` with `μ = X_μβ_μ + o_μ`: `mu` and `sigma` are the per-row
/// mean (offset included) and standard deviation, in the units `y` is in.
pub(crate) struct GaussianLocationScaleMeanFit<'a> {
    pub beta_mu: ArrayView1<'a, f64>,
    pub y: ArrayView1<'a, f64>,
    pub prior_weights: ArrayView1<'a, f64>,
    pub mu: ArrayView1<'a, f64>,
    pub sigma: ArrayView1<'a, f64>,
}

/// The variance-component test of every mean-block term of a Gaussian
/// location-scale fit (`None`: the fit has no evaluable mean block, and every
/// term carries the missing row state as its reason).
///
/// Conditional on the fitted shape of `σ̂`, the mean block is the weighted
/// Gaussian regression `yᵢ ~ N(μᵢ, φσ̂ᵢ²)`: Fisher weight `wᵢ/σ̂ᵢ²` and score
/// `wᵢ(yᵢ − μ̂ᵢ)/σ̂ᵢ²`. The mean/log-σ block of the Fisher information is zero
/// (`E[∂²ℓ/∂μ∂σ] = 0`), so the log-σ coefficients need no projection and the
/// mean block's own score is its efficient score.
///
/// The level `φ` is estimated, not taken as 1. `σ̂` is fitted to the same
/// residuals the score is built from, so the standardized residuals are
/// smaller than unit variance by the degrees of freedom the mean fit spent.
/// Taking `φ = 1` would ignore that and read the score too large. Treating
/// `φ` as estimated refers the score to the residual `χ²_ν` of the
/// standardized model, which is its exact law given the shape.
pub(crate) fn gaussian_location_scale_variance_component_test_records(
    mean_design: &gam_terms::smooth::TermCollectionDesign,
    fitted: Option<GaussianLocationScaleMeanFit<'_>>,
) -> Vec<gam_terms::inference::variance_component_test::VarianceComponentTestRecord> {
    use gam_terms::inference::variance_component_test::{
        VarianceComponentTestScale, VarianceComponentTestUnavailable,
    };

    let n_rows = mean_design.design.nrows();
    let scale = VarianceComponentTestScale::Estimated;
    let Some(fitted) = fitted else {
        return variance_component_test_records_from_rows(
            mean_design,
            ArrayView1::from(&[][..]),
            Err(VarianceComponentTestUnavailable::NoIrlsRowState),
            scale,
        );
    };
    let rows_agree = [fitted.y.len(), fitted.prior_weights.len(), fitted.mu.len(), fitted.sigma.len()]
        .iter()
        .all(|&len| len == n_rows);
    let rows = if rows_agree
        && fitted.sigma.iter().all(|s| s.is_finite() && *s > 0.0)
        && fitted.prior_weights.iter().all(|w| w.is_finite() && *w >= 0.0)
    {
        let weights = Array1::from_shape_fn(n_rows, |i| {
            fitted.prior_weights[i] / (fitted.sigma[i] * fitted.sigma[i])
        });
        let score = Array1::from_shape_fn(n_rows, |i| weights[i] * (fitted.y[i] - fitted.mu[i]));
        score
            .iter()
            .all(|value| value.is_finite())
            .then(|| ScoreRowState {
                hessian_weights: weights.clone(),
                score_weights: weights,
                score,
                linear_predictor: fitted.mu.to_owned(),
            })
            .ok_or(VarianceComponentTestUnavailable::NoIrlsRowState)
    } else {
        Err(VarianceComponentTestUnavailable::NoIrlsRowState)
    };
    variance_component_test_records_from_rows(mean_design, fitted.beta_mu, rows, scale)
}

/// One record per tested term of `design`, scored from the fit's row state
/// (or carrying the reason that state is unavailable).
fn variance_component_test_records_from_rows(
    design: &gam_terms::smooth::TermCollectionDesign,
    beta: ArrayView1<'_, f64>,
    rows: Result<
        ScoreRowState,
        gam_terms::inference::variance_component_test::VarianceComponentTestUnavailable,
    >,
    scale: gam_terms::inference::variance_component_test::VarianceComponentTestScale,
) -> Vec<gam_terms::inference::variance_component_test::VarianceComponentTestRecord> {
    use gam_terms::inference::variance_component_test::{
        VarianceComponentTermRequest, VarianceComponentTestBasis, VarianceComponentTestInput,
        VarianceComponentTestOutcome, VarianceComponentTestRecord,
        VarianceComponentTestUnavailable,
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

    let rows_state = match rows {
        Ok(rows_state) => rows_state,
        Err(reason) => return unavailable(reason),
    };
    if beta.len() != design.design.ncols() {
        return unavailable(VarianceComponentTestUnavailable::DesignUnavailable);
    }
    let basis = match VarianceComponentTestBasis::new(VarianceComponentTestInput {
        design: &design.design,
        beta,
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
/// on exactly its range; a range no penalty covers is refused.
/// A smooth is tested when
/// [`gam_terms::smooth::TermCollectionDesign::smooth_variance_component_penalties`]
/// admits it; every other smooth keeps the Wald test.
fn variance_component_test_terms(
    design: &gam_terms::smooth::TermCollectionDesign,
) -> Vec<(
    String,
    gam_terms::inference::variance_component_test::VarianceComponentTermRequest,
)> {
    use gam_terms::inference::variance_component_test::VarianceComponentTermRequest;

    let mut terms = Vec::new();
    for (name, range) in &design.random_effect_ranges {
        let penalties: Vec<Array2<f64>> = design
            .penalties
            .iter()
            .filter(|penalty| penalty.col_range == *range)
            .map(|penalty| penalty.local.clone())
            .collect();
        terms.push((
            name.clone(),
            VarianceComponentTermRequest {
                range: range.clone(),
                penalties,
            },
        ));
    }
    for term in &design.smooth.terms {
        if let Some(penalties) = design.smooth_variance_component_penalties(term) {
            terms.push((
                term.name.clone(),
                VarianceComponentTermRequest {
                    range: design.smooth_global_range(term),
                    penalties,
                },
            ));
        }
    }
    terms
}
