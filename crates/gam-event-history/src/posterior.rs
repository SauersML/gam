//! Posterior-predictive event-history prediction: the average of the final
//! probabilities over the global parameter posterior (gam#2964, SPEC 3).
//!
//! [`super::forecast`]'s probabilities are conditional on the fitted
//! parameter state `θ̂`. The posterior-predictive probability averages the
//! FINAL probability over the posterior of the global parameters:
//!
//! ```text
//! P(· | data) = E_θ[ P(· | data, θ) ]
//! ```
//!
//! The probabilities are nonlinear in `θ`, so this is not `P(· | data, θ̂)`
//! and no summary of the posterior stands in for it. Averaging the loading
//! covariance, averaging the baseline, or evaluating a probability at
//! averaged coefficients each answer a different question — the contract the
//! deleted `joint/coefficient_prediction.rs` stated for the joint law, kept
//! here for the marginal one. So every state the average visits carries its
//! own reference evolution (a [`RiskSetCentring`] rebuilt at those
//! coefficients, never the fitted snapshot) and its own history-conditioned
//! latent law (the filter rerun at them), and only the probabilities those
//! produce are averaged.
//!
//! The controlled approximation is a Gauss-Hermite product rule over the
//! Gaussian approximation to the posterior, in the coordinates that posterior
//! resolves. Its resolution is the crate's existing one and needs no new
//! concept: the average is taken on the fit's own Gauss-Hermite order and
//! again on its next rung `2G − 1`, exactly as a forecast window already
//! checks its latent quadrature, the gap is returned beside every
//! probability, and a gap the fit's `quadrature_tolerance` does not cover is
//! a typed refusal rather than a number.
//!
//! ## What this costs, and where it refuses
//!
//! The rule has `G^d` points on the fit's order and `(2G − 1)^d` on the check
//! order, for `d` the number of posterior directions it integrates over, and
//! each point is a reference rebuild followed by a full prediction. `d` is
//! the number of directions the posterior covariance resolves, so a fit with
//! many coefficients has a rule with more points than anyone will run. There
//! is no smaller rule here that is still checked, so the states a rule holds
//! are charged against this machine's materialisation budget and a rule that
//! does not fit it is REFUSED, with `d`, the order and the point count named.
//! Refusing is the answer: a plug-in probability relabelled as a
//! posterior-predictive one is the defect this module exists to remove.
//!
//! Reducing `d` is separate, stated work. The screen that does it keeps the
//! leading directions until the omitted subspace's standard deviation of
//! `η⁰`, over the window's and the reference population's design rows and at
//! the rule's reach, falls under `quadrature_tolerance` nats. The omission is
//! then bounded rather than assumed, because a killed-process functional is
//! monotone in its intensities: scaling every intensity by `e^{±δ}` brackets
//! the survival between `S^{e^{δ}}` and `S^{e^{−δ}}`, whose width is at most
//! `δ/e` since `max_x x ln(1/x) = 1/e`.

use super::chain::{GaussHermite, product_grid_size};
use super::cohort::EventHistoryError;
use super::covariance::eigenmodes;
use super::family::{EventHistoryFit, RiskSetCentring, risk_set_normaliser_of};
use super::forecast::Forecast;
use super::predictor::PredictionModel;
use ndarray::{Array1, Array2};

/// One global parameter state a prediction conditions on: the per-mark
/// coefficient blocks, the latent loadings and log-rates, and the reference
/// evolution those coefficients imply.
///
/// The fitted state and a state the posterior rule visits are the same kind
/// of object, read by the same code, so no prediction path can quietly pair a
/// perturbed coefficient with the fitted reference law.
#[derive(Clone, Debug)]
pub(crate) struct ParameterState {
    /// Per mark, that block's coefficients.
    pub mark_betas: Vec<Array1<f64>>,
    /// Loadings, index `d * atoms + k`.
    pub loadings: Vec<f64>,
    /// The log of each atom's dimensionless rate. This is the coordinate the
    /// fit reports and every consumer exponentiates, so a state built at the
    /// fitted coefficients carries the fit's own `log_rates` bit for bit.
    pub log_rates: Vec<f64>,
    /// The reference evolution at these coefficients, or `None` where the
    /// baselines are centred on the stationary prior.
    pub centring: Option<RiskSetCentring>,
}

impl ParameterState {
    /// The dimensionless rates `ν` the filter takes.
    pub(crate) fn rates(&self) -> Vec<f64> {
        self.log_rates.iter().map(|r| r.exp()).collect()
    }

    /// The state the fit itself is at: the fit's reported loadings and
    /// log-rates and the reference snapshot it returned, so a conditional
    /// prediction built from this is the prediction that was made before this
    /// module existed, bit for bit.
    pub(crate) fn fitted(fit: &EventHistoryFit) -> Self {
        let marks = fit.marks();
        Self {
            mark_betas: (0..marks).map(|d| fit.mark_coefficients(d).clone()).collect(),
            loadings: Self::fitted_loadings(fit),
            log_rates: fit.log_rates.clone(),
            centring: fit.centring.clone(),
        }
    }

    /// The fit's reported loadings flattened as the filter indexes them,
    /// `d * atoms + k`. A saved predictor records them through this, so an
    /// artifact and the fit it came from carry one vector, not two orderings
    /// (`super::predictor`).
    pub(crate) fn fitted_loadings(fit: &EventHistoryFit) -> Vec<f64> {
        let (marks, atoms) = (fit.marks(), fit.rank());
        let mut loadings = Vec::with_capacity(marks * atoms);
        for d in 0..marks {
            for k in 0..atoms {
                loadings.push(fit.loadings[[d, k]]);
            }
        }
        loadings
    }

    /// The state at an arbitrary coefficient vector, in the model's layout.
    /// The latent block is read through the model's own rate chart and the
    /// reference evolution is rebuilt from these coefficients, never carried
    /// over from the fitted state.
    pub(crate) fn at(
        model: &PredictionModel<'_>,
        coefficients: &[f64],
    ) -> Result<Self, EventHistoryError> {
        let (marks, atoms) = (model.marks, model.atoms);
        let offsets = &model.block_offsets;
        let width = model.total_width();
        if coefficients.len() != width {
            return Err(EventHistoryError::InvalidInput {
                reason: format!(
                    "a parameter state needs {width} coefficients in the model's layout, got {}",
                    coefficients.len()
                ),
            });
        }
        let mark_betas: Vec<Array1<f64>> = (0..marks)
            .map(|d| Array1::from(coefficients[offsets[d]..offsets[d + 1]].to_vec()))
            .collect();
        let latent = Array1::from(coefficients[offsets[marks]..].to_vec());
        let loadings = coefficients[offsets[marks]..offsets[marks] + marks * atoms].to_vec();
        // The two lines the fit's own latent report runs, so a state at the
        // fitted coefficients reproduces the fit's loadings and log-rates.
        let log_rates: Vec<f64> = model
            .atom_rates(&latent)
            .iter()
            .map(|nu| nu.ln())
            .collect();
        // The reference evolution is a function of the coefficients alone, so
        // at the model's own coefficients it IS the snapshot it carries.
        // Reading it back rather than recomputing it keeps a prediction at
        // `θ̂` identical to the conditional one bit for bit, and spares the
        // rule's centre point a reference rebuild.
        let centring = if coefficients == model.fitted_coefficients.as_slice() {
            model.fitted.centring.clone()
        } else {
            model.reference_at_coefficients(coefficients)?
        };
        Ok(Self {
            mark_betas,
            loadings,
            log_rates,
            centring,
        })
    }

    /// `log M_d(t)` for one stratum under this state's reference evolution,
    /// or an empty vector where the baselines are centred on the stationary
    /// prior: [`EventHistoryFit::risk_set_normaliser_at`] read at a state
    /// rather than at the fit.
    pub(crate) fn risk_set_normaliser_at(
        &self,
        marks: usize,
        stratum: usize,
        t: f64,
    ) -> Result<Vec<f64>, EventHistoryError> {
        risk_set_normaliser_of(self.centring.as_ref(), marks, stratum, t)
    }
}

/// A posterior-predictive prediction: the averaged probabilities, the window
/// errors they inherit, and the posterior rule's own error beside each one.
///
/// `survival` and `expected_counts` are `E_θ[·]` under the rule. The `_error`
/// fields are the weighted averages of each state's own window refinement
/// error, which bound the average of the window quadratures. The
/// `_posterior_gap` fields are the distance to the same average on the next
/// Gauss-Hermite rung: they measure the rule itself, which no field of a
/// conditional [`Forecast`] does.
#[derive(Clone, Debug)]
pub struct PosteriorPredictiveForecast {
    pub horizons: Vec<f64>,
    pub survival: Vec<f64>,
    pub expected_counts: Array2<f64>,
    pub survival_error: Vec<f64>,
    pub expected_count_errors: Array2<f64>,
    /// Per horizon, `|average at `order` − average at `check_order`|` for the
    /// survival.
    pub survival_posterior_gap: Vec<f64>,
    /// The same for every entry of `expected_counts`.
    pub expected_count_posterior_gaps: Array2<f64>,
    /// How many posterior directions the rule integrated over.
    pub directions: usize,
    /// The posterior variance the decomposition's own resolution left out,
    /// summed over the eigenvalues below that floor. It is the spread no rule
    /// here integrates over, reported so it is not mistaken for zero.
    pub unresolved_posterior_variance: f64,
    /// The Gauss-Hermite order of the returned average, and the rung its gap
    /// was measured against.
    pub order: usize,
    pub check_order: usize,
    /// The parameter states each rule visited.
    pub states: usize,
    pub check_states: usize,
}

/// The posterior directions a rule integrates over: a step in coefficient
/// space per direction, one posterior standard deviation long.
///
/// The directions are the eigenvectors of the posterior covariance, and a
/// direction is kept only while its eigenvalue stands above the
/// decomposition's own resolution. A symmetric eigensolver returns
/// eigenvalues with a backward error of order `p · ε · λ_max` on a `p × p`
/// matrix, so below that an eigenvalue is not a variance the decomposition
/// resolves and its direction carries no spread that can be integrated. An
/// exactly singular posterior therefore yields no direction at all and the
/// rule is the single fitted state, which is the conditional prediction.
struct PosteriorDirections {
    /// `√λ_k · v_k` per kept direction.
    steps: Vec<Vec<f64>>,
    /// The fitted coefficients the steps are taken from.
    centre: Vec<f64>,
    /// The variance the resolution floor left out.
    unresolved_variance: f64,
}

impl PosteriorDirections {
    /// The directions of a model's own posterior.
    ///
    /// The covariance is the one `predict()` reads everywhere else in this
    /// repository: the smoothing-corrected `V_c` when the fit published one,
    /// which also propagates the uncertainty in `λ̂`, falling back to the
    /// conditional `V_b` only where the fit has no smoothing coordinate to
    /// correct for. A model that publishes neither states no posterior over
    /// its coefficients, and averaging over a posterior that does not exist is
    /// a typed refusal — which a saved predictor inherits, because the
    /// covariance is what its document carries as the posterior
    /// representation.
    fn of(model: &PredictionModel<'_>) -> Result<Self, EventHistoryError> {
        let centre = model.fitted_coefficients.clone();
        let width = centre.len();
        let covariance = model
            .posterior_covariance
            .filter(|c| c.nrows() == width && c.ncols() == width)
            .ok_or_else(|| EventHistoryError::Fit {
                reason: format!(
                    "the model publishes no {width}×{width} posterior covariance of its coefficients, so it states no posterior for a prediction to be averaged over"
                ),
            })?;
        let (values, vectors) =
            eigenmodes(covariance, gam_linalg::roundoff::SymmetricAssembly::Mirrored)?;
        if let Some(bad) = values.iter().position(|v| !v.is_finite()) {
            return Err(EventHistoryError::NumericalFailure {
                reason: format!(
                    "eigenvalue {bad} of the posterior covariance is {}",
                    values[bad]
                ),
            });
        }
        let largest = values.iter().fold(0.0_f64, |a, &b| a.max(b));
        let floor = width as f64 * f64::EPSILON * largest;
        let mut steps = Vec::new();
        let mut unresolved_variance = 0.0;
        for (k, &value) in values.iter().enumerate() {
            if value > floor {
                let scale = value.sqrt();
                steps.push((0..width).map(|q| scale * vectors[[q, k]]).collect());
            } else {
                unresolved_variance += value.max(0.0);
            }
        }
        Ok(Self {
            steps,
            centre,
            unresolved_variance,
        })
    }

    fn count(&self) -> usize {
        self.steps.len()
    }

    /// The coefficient vector at a rule point, given the point's abscissa in
    /// posterior standard deviations along each direction.
    fn coefficients_at(&self, abscissae: &[f64]) -> Vec<f64> {
        let mut out = self.centre.clone();
        for (step, &a) in self.steps.iter().zip(abscissae.iter()) {
            for (q, value) in out.iter_mut().enumerate() {
                *value += a * step[q];
            }
        }
        out
    }
}

/// A Gauss-Hermite product rule over the posterior directions: the abscissae
/// in posterior standard deviations and the standard-normal weight of every
/// point, direction 0 fastest.
struct ProductRule {
    order: usize,
    directions: usize,
    /// `√2 · x_l`, the abscissa of node `l` in standard deviations.
    abscissae: Vec<f64>,
    /// `w_l / √π`, the standard-normal expectation's weight of node `l`.
    weights: Vec<f64>,
}

impl ProductRule {
    fn new(order: usize, directions: usize) -> Result<Self, EventHistoryError> {
        let gh = GaussHermite::new(order)?;
        Ok(Self {
            order,
            directions,
            abscissae: gh
                .nodes
                .iter()
                .map(|x| std::f64::consts::SQRT_2 * x)
                .collect(),
            weights: gh.normal_weights.clone(),
        })
    }

    /// `order^directions`, refused rather than wrapped.
    fn points(&self) -> Result<usize, EventHistoryError> {
        product_grid_size(self.order, self.directions)
    }

    /// The abscissae and the product weight of flat point `p`, direction 0
    /// fastest.
    fn point(&self, p: usize) -> (Vec<f64>, f64) {
        let mut rest = p;
        let mut abscissae = Vec::with_capacity(self.directions);
        let mut weight = 1.0;
        for _ in 0..self.directions {
            let l = rest % self.order;
            rest /= self.order;
            abscissae.push(self.abscissae[l]);
            weight *= self.weights[l];
        }
        (abscissae, weight)
    }

    /// Refuse a rule whose parameter states would not fit this machine's
    /// materialisation budget, naming what it would have taken.
    ///
    /// This is a resource guard: it refuses the rule and never trims it to a
    /// smaller one, because a rule chosen by a budget would make the reported
    /// probability a function of the machine it ran on.
    fn fits(&self, model: &PredictionModel<'_>) -> Result<usize, EventHistoryError> {
        let points = self.points()?;
        let per_state = model.total_width()
            + model.fitted.centring.as_ref().map_or(0, |snapshot| {
                snapshot.log_normaliser.len()
                    + snapshot.log_risk_mass.len()
                    + snapshot.profiles.len()
            });
        let bytes = points as f64 * per_state as f64 * std::mem::size_of::<f64>() as f64;
        let budget = gam_runtime::resource::ResourcePolicy::default_library()
            .max_single_materialization_bytes as f64;
        if bytes > budget {
            return Err(EventHistoryError::NumericalFailure {
                reason: format!(
                    "a posterior rule of order {} over {} resolved posterior directions has {points} points, whose parameter states and reference evolutions need about {:.1} GiB, above this machine's {:.1} GiB materialisation budget; the average is refused rather than taken on a smaller rule, which would make the probability a function of the machine it ran on",
                    self.order,
                    self.directions,
                    bytes / f64::from(1u32 << 30),
                    budget / f64::from(1u32 << 30)
                ),
            });
        }
        Ok(points)
    }
}

/// The running weighted average of a rule's predictions.
struct Average {
    horizons: Vec<f64>,
    survival: Vec<f64>,
    survival_error: Vec<f64>,
    counts: Array2<f64>,
    count_errors: Array2<f64>,
    weight: f64,
}

impl Average {
    fn new(horizons: &[f64], marks: usize) -> Self {
        Self {
            horizons: horizons.to_vec(),
            survival: vec![0.0; horizons.len()],
            survival_error: vec![0.0; horizons.len()],
            counts: Array2::zeros((horizons.len(), marks)),
            count_errors: Array2::zeros((horizons.len(), marks)),
            weight: 0.0,
        }
    }

    fn add(&mut self, forecast: &Forecast, weight: f64) -> Result<(), EventHistoryError> {
        if forecast.horizons != self.horizons {
            return Err(EventHistoryError::NumericalFailure {
                reason: "a parameter state returned a prediction on different horizons".to_string(),
            });
        }
        for (i, value) in forecast.survival.iter().enumerate() {
            self.survival[i] += weight * value;
            self.survival_error[i] += weight * forecast.survival_error[i];
        }
        self.counts.scaled_add(weight, &forecast.expected_counts);
        self.count_errors
            .scaled_add(weight, &forecast.expected_count_errors);
        self.weight += weight;
        Ok(())
    }

    /// Divide through by the weight the rule actually carried. Gauss-Hermite
    /// weights sum to one only to their own rounding, and a probability that
    /// is an average must be an average under a probability.
    fn finish(mut self) -> Self {
        let inverse = 1.0 / self.weight;
        for value in self.survival.iter_mut() {
            *value *= inverse;
        }
        for value in self.survival_error.iter_mut() {
            *value *= inverse;
        }
        self.counts *= inverse;
        self.count_errors *= inverse;
        self
    }
}

/// Run one rule and average what it returns.
fn average_over<F>(
    directions: &PosteriorDirections,
    rule: &ProductRule,
    points: usize,
    horizons: &[f64],
    marks: usize,
    at: &F,
) -> Result<Average, EventHistoryError>
where
    F: Fn(&[f64]) -> Result<Forecast, EventHistoryError>,
{
    let mut average = Average::new(horizons, marks);
    for p in 0..points {
        let (abscissae, weight) = rule.point(p);
        let forecast = at(&directions.coefficients_at(&abscissae))?;
        average.add(&forecast, weight)?;
    }
    if !(average.weight > 0.0) {
        return Err(EventHistoryError::NumericalFailure {
            reason: format!(
                "the posterior rule of order {} over {} directions carried total weight {}",
                rule.order, rule.directions, average.weight
            ),
        });
    }
    Ok(average.finish())
}

/// The bar a posterior rule's own error is read against, and what it names
/// when it refuses.
struct PosteriorBar {
    tolerance: f64,
    directions: usize,
    order: usize,
    check_order: usize,
}

impl PosteriorBar {
    /// Refuse a posterior rule whose own error is not covered by the accuracy
    /// the fit certifies.
    ///
    /// The bar is `tolerance` of the quantity's own scale, and the scale is
    /// the larger of the returned value and the window error already reported
    /// beside it. Reading it against the value alone would refuse a
    /// probability that is small because it is small; reading it against the
    /// window error alone would accept a rule whose error dominates a
    /// quantity the window resolved sharply. The larger of the two says both:
    /// the rule must be within the certified relative accuracy of the
    /// quantity, and it must not become the dominant error of a quantity
    /// whose other error is already bounded.
    fn resolved(
        &self,
        gap: f64,
        value: f64,
        window_error: f64,
        what: &str,
    ) -> Result<(), EventHistoryError> {
        let scale = value.abs().max(window_error.abs());
        if !(gap <= self.tolerance * scale) {
            return Err(EventHistoryError::NumericalFailure {
                reason: format!(
                    "the posterior average of {what} moves by {gap:.3e} between Gauss-Hermite orders {} and {} over {} posterior directions, past the {:.3e} the fit's quadrature tolerance {:.3e} allows on a quantity of scale {scale:.3e}; the posterior rule does not resolve this prediction",
                    self.order,
                    self.check_order,
                    self.directions,
                    self.tolerance * scale,
                    self.tolerance
                ),
            });
        }
        Ok(())
    }
}

/// The posterior-predictive average of a prediction, on the fit's own
/// Gauss-Hermite order and on its next rung, with the gap between them
/// returned and checked.
///
/// `at` runs the whole prediction at one coefficient vector: it must rebuild
/// that state's reference evolution and rerun that state's filter, which
/// [`ParameterState::at`] does, and it must read nothing from the fit that
/// the coefficients determine.
pub(crate) fn posterior_predictive<F>(
    model: &PredictionModel<'_>,
    horizons: &[f64],
    at: F,
) -> Result<PosteriorPredictiveForecast, EventHistoryError>
where
    F: Fn(&[f64]) -> Result<Forecast, EventHistoryError>,
{
    let marks = model.marks;
    let directions = PosteriorDirections::of(model)?;
    let order = model.gh.order;
    let bar = PosteriorBar {
        tolerance: model.quadrature_tolerance,
        directions: directions.count(),
        order,
        check_order: 2 * order - 1,
    };
    let rule = ProductRule::new(bar.order, directions.count())?;
    let check = ProductRule::new(bar.check_order, directions.count())?;
    let states = rule.fits(model)?;
    let check_states = check.fits(model)?;
    let averaged = average_over(&directions, &rule, states, horizons, marks, &at)?;
    let checked = average_over(&directions, &check, check_states, horizons, marks, &at)?;

    let mut survival_posterior_gap = vec![0.0; horizons.len()];
    for (i, gap) in survival_posterior_gap.iter_mut().enumerate() {
        *gap = (averaged.survival[i] - checked.survival[i]).abs();
        bar.resolved(
            *gap,
            averaged.survival[i],
            averaged.survival_error[i],
            &format!("the survival at horizon {}", horizons[i]),
        )?;
    }
    let mut expected_count_posterior_gaps = Array2::<f64>::zeros((horizons.len(), marks));
    for i in 0..horizons.len() {
        for d in 0..marks {
            let gap = (averaged.counts[[i, d]] - checked.counts[[i, d]]).abs();
            expected_count_posterior_gaps[[i, d]] = gap;
            bar.resolved(
                gap,
                averaged.counts[[i, d]],
                averaged.count_errors[[i, d]],
                &format!("the expected count of mark {d} at horizon {}", horizons[i]),
            )?;
        }
    }
    Ok(PosteriorPredictiveForecast {
        horizons: averaged.horizons,
        survival: averaged.survival,
        expected_counts: averaged.counts,
        survival_error: averaged.survival_error,
        expected_count_errors: averaged.count_errors,
        survival_posterior_gap,
        expected_count_posterior_gaps,
        directions: directions.count(),
        unresolved_posterior_variance: directions.unresolved_variance,
        order: bar.order,
        check_order: bar.check_order,
        states,
        check_states,
    })
}
