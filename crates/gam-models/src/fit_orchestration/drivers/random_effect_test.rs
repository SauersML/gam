// The per-term variance-component test — the driver that turns a fitted GAM
// into "does each `group()` block carry a between-group effect?" and, on the
// standard route, "does each ridged linear term carry any effect?" (#3573).
//
// `include!`d into `drivers/mod.rs` like the other self-contained inference
// subsystems, so it shares the driver's flat namespace and import surface.
//
// The statistic and its exact reference law live in
// `gam_terms::inference::random_effect_test`. This file only reads a fit's row
// state (the retained IRLS state of a standard fit, or the mean block's own
// Fisher weights and score of a location-scale fit) and the realized
// coefficient layout and hands them over; every term gets a record, and a term
// the test cannot score carries the typed reason instead of a p-value.

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
/// each such row.
pub fn variance_component_test_records(
    design: &gam_terms::smooth::TermCollectionDesign,
    fit: &UnifiedFitResult,
) -> VarianceComponentTestRecords {
    use gam_terms::inference::random_effect_test::{
        RandomEffectTestRecord, RandomEffectTestScale, RandomEffectTestUnavailable,
    };

    let random_effect_count = design.random_effect_ranges.len();
    let ranges: Vec<(String, Range<usize>)> = design
        .random_effect_ranges
        .iter()
        .cloned()
        .chain(design.ridged_linear_ranges())
        .collect();
    let split = |mut records: Vec<RandomEffectTestRecord>| {
        let linear_term = records.split_off(random_effect_count);
        VarianceComponentTestRecords {
            random_effect: records,
            linear_term,
        }
    };
    if ranges.is_empty() {
        return split(Vec::new());
    }
    let unavailable =
        |reason: RandomEffectTestUnavailable| split(unavailable_records(&ranges, 0, reason));

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
    split(scored_records(
        &design.design,
        &ranges,
        0,
        fit.beta.view(),
        &rows_state,
        scale,
    ))
}

/// The fitted state of a Gaussian location-scale mean block,
/// `yᵢ ~ N(μᵢ, σᵢ²)` with `μ = X_μβ_μ + o_μ`: `mu` and `sigma` are the per-row
/// mean (offset included) and standard deviation, in the units `y` is in.
pub(crate) struct GaussianLocationScaleMeanFit<'a> {
    pub beta_mu: ArrayView1<'a, f64>,
    pub y: ArrayView1<'a, f64>,
    pub prior_weights: ArrayView1<'a, f64>,
    pub mu: ArrayView1<'a, f64>,
    pub sigma: ArrayView1<'a, f64>,
}

/// The variance-component test of every random-effect block of a Gaussian
/// location-scale fit's mean predictor. `coefficient_offset` is where the mean
/// block starts in the fit's flat coefficient layout, so each record carries
/// its GLOBAL range; `fitted` is the mean block's fitted state, or the typed
/// reason the route has none.
///
/// Conditional on the fitted shape of `σ̂`, the mean block is the weighted
/// Gaussian regression `yᵢ ~ N(μᵢ, φσ̂ᵢ²)` with identity link: Fisher and
/// curvature weight `wᵢ/σ̂ᵢ²`, working residual `yᵢ − μ̂ᵢ`, score
/// `wᵢ(yᵢ − μ̂ᵢ)/σ̂ᵢ²`. The mean/log-σ block of the Fisher information is zero,
/// `E[∂²ℓᵢ/∂μᵢ∂ησᵢ] = −2wᵢ E[yᵢ − μᵢ]/σᵢ² · ∂σᵢ/∂ησᵢ / σᵢ = 0`, so the
/// log-σ coefficients need no projection: the mean block's own score is its
/// efficient score, and projecting the mean predictor's other columns out in
/// the `wᵢ/σ̂ᵢ²` metric is the whole nuisance adjustment.
///
/// The level `φ` is estimated, not taken as 1. `σ̂` is fitted to the same
/// residuals the score is built from, so the standardized residuals
/// `(yᵢ − μ̂ᵢ)/σ̂ᵢ` fall short of unit variance by what the mean fit spent.
/// Taking `φ = 1` would read the score against too large a variance and be
/// conservative. Treating `φ` as estimated refers the score to the residual
/// `χ²_ν` of the standardized unpenalized model, its exact law given the shape.
/// The ratio is also invariant to the response's units, so the test is the
/// same whether the fit reports raw or standardized quantities.
///
/// This route scores the mean design's `group()` blocks only. A ridged linear
/// term in a mean predictor is the same variance component #3573 names, but the
/// location-scale artifacts carry ONE record list, with no
/// [`VarianceComponentTestRecords`] split to report a linear term in; scoring
/// one here would file it as a random effect.
pub(crate) fn gaussian_location_scale_random_effect_test_records(
    mean_design: &gam_terms::smooth::TermCollectionDesign,
    coefficient_offset: usize,
    fitted: Result<
        GaussianLocationScaleMeanFit<'_>,
        gam_terms::inference::random_effect_test::RandomEffectTestUnavailable,
    >,
) -> Vec<gam_terms::inference::random_effect_test::RandomEffectTestRecord> {
    use gam_terms::inference::random_effect_test::{
        RandomEffectTestScale, RandomEffectTestUnavailable,
    };

    if mean_design.random_effect_ranges.is_empty() {
        return Vec::new();
    }
    let fitted = match fitted {
        Ok(fitted) => fitted,
        Err(reason) => {
            return random_effect_unavailable_records(mean_design, coefficient_offset, reason);
        }
    };
    let n_rows = mean_design.design.nrows();
    let rows_agree = [
        fitted.y.len(),
        fitted.prior_weights.len(),
        fitted.mu.len(),
        fitted.sigma.len(),
    ]
    .iter()
    .all(|&len| len == n_rows);
    let rows_finite = rows_agree
        && fitted.y.iter().all(|v| v.is_finite())
        && fitted.mu.iter().all(|v| v.is_finite())
        && fitted.sigma.iter().all(|s| s.is_finite() && *s > 0.0)
        && fitted
            .prior_weights
            .iter()
            .all(|w| w.is_finite() && *w >= 0.0);
    if !rows_finite {
        return random_effect_unavailable_records(
            mean_design,
            coefficient_offset,
            RandomEffectTestUnavailable::DesignUnavailable,
        );
    }
    let weights = Array1::from_shape_fn(n_rows, |i| {
        fitted.prior_weights[i] / (fitted.sigma[i] * fitted.sigma[i])
    });
    let score = Array1::from_shape_fn(n_rows, |i| weights[i] * (fitted.y[i] - fitted.mu[i]));
    let rows_state = ScoreRowState {
        hessian_weights: weights.clone(),
        score_weights: weights,
        score,
        linear_predictor: fitted.mu.to_owned(),
    };
    scored_records(
        &mean_design.design,
        &mean_design.random_effect_ranges,
        coefficient_offset,
        fitted.beta_mu,
        &rows_state,
        RandomEffectTestScale::Estimated,
    )
}

/// One record per random-effect block of `design`, each carrying `reason`.
/// Location-scale routes whose predictor has no row state for the score use
/// this so every `group()` row states why it has no p-value.
pub(crate) fn random_effect_unavailable_records(
    design: &gam_terms::smooth::TermCollectionDesign,
    coefficient_offset: usize,
    reason: gam_terms::inference::random_effect_test::RandomEffectTestUnavailable,
) -> Vec<gam_terms::inference::random_effect_test::RandomEffectTestRecord> {
    unavailable_records(&design.random_effect_ranges, coefficient_offset, reason)
}

/// One record per range in `ranges`, each carrying `reason`. `ranges` is the
/// term list the calling route scores, so a route reports a reason for exactly
/// the terms it would otherwise have scored.
fn unavailable_records(
    ranges: &[(String, Range<usize>)],
    coefficient_offset: usize,
    reason: gam_terms::inference::random_effect_test::RandomEffectTestUnavailable,
) -> Vec<gam_terms::inference::random_effect_test::RandomEffectTestRecord> {
    use gam_terms::inference::random_effect_test::{
        RandomEffectTestOutcome, RandomEffectTestRecord,
    };
    ranges
        .iter()
        .map(|(name, range)| RandomEffectTestRecord {
            term: name.clone(),
            coefficient_range: (range.start + coefficient_offset)..(range.end + coefficient_offset),
            outcome: RandomEffectTestOutcome::Unavailable { reason },
        })
        .collect()
}

/// Score every term of `ranges` against one shared basis built from `design`
/// and `rows_state`. `beta` is the coefficient vector of `design`'s own
/// columns; `coefficient_offset` places the design's block in the fit's flat
/// layout for the records. The one scoring body: the standard route passes its
/// `group()` blocks followed by its ridged linear terms, a location-scale route
/// passes its `group()` blocks, and neither can drift from the other.
fn scored_records(
    design: &DesignMatrix,
    ranges: &[(String, Range<usize>)],
    coefficient_offset: usize,
    beta: ArrayView1<'_, f64>,
    rows_state: &ScoreRowState,
    scale: gam_terms::inference::random_effect_test::RandomEffectTestScale,
) -> Vec<gam_terms::inference::random_effect_test::RandomEffectTestRecord> {
    use gam_terms::inference::random_effect_test::{
        RandomEffectTermRequest, RandomEffectTestBasis, RandomEffectTestInput,
        RandomEffectTestOutcome, RandomEffectTestRecord, RandomEffectTestUnavailable,
    };

    if beta.len() != design.ncols() {
        return unavailable_records(
            ranges,
            coefficient_offset,
            RandomEffectTestUnavailable::DesignUnavailable,
        );
    }
    let basis = match RandomEffectTestBasis::new(RandomEffectTestInput {
        design,
        beta,
        hessian_weights: rows_state.hessian_weights.view(),
        score_weights: rows_state.score_weights.view(),
        score: rows_state.score.view(),
        scale,
    }) {
        Ok(basis) => basis,
        Err(reason) => return unavailable_records(ranges, coefficient_offset, reason),
    };
    let requests: Vec<RandomEffectTermRequest> = ranges
        .iter()
        .map(|(_, range)| RandomEffectTermRequest {
            range: range.clone(),
        })
        .collect();
    ranges
        .iter()
        .zip(basis.test_terms(&requests))
        .map(|((name, range), outcome)| RandomEffectTestRecord {
            term: name.clone(),
            coefficient_range: (range.start + coefficient_offset)..(range.end + coefficient_offset),
            outcome: RandomEffectTestOutcome::from(outcome),
        })
        .collect()
}
