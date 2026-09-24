//! The single-event survival surfaces under an inequality-truncated coefficient posterior
//! (gam#3038, gam#3575): the location-scale and Royston-Parmar fits whose posterior is
//! `N(β_unc, Σ)` truncated to the fit's cone. Every node of the law's joint rule is a feasible
//! coefficient vector, the surfaces are replayed at it, and each cell's moments are certified on
//! the spread of the rule's replicate lattices ([`truncated_survival_surface_moments`]).

use super::*;

/// The cone-truncated coefficient posterior of a location-scale or
/// Royston-Parmar fit under `covariance_mode`, as a rule over whole coefficient
/// vectors, or `None` when the selected law retains no constraint row.
pub(super) fn truncated_survival_posterior_draws(
    model: &SavedModel,
    covariance_mode: SurvivalPredictionCovarianceMode,
) -> Result<Option<TruncatedCoefficientDraws>, SurvivalPredictError> {
    let fit = fit_result_from_saved_model_for_prediction(model)?;
    fit.require_posterior_mean("survival posterior-mean prediction")
        .map_err(|error| SurvivalPredictError::PosteriorCovariance {
            reason: error.to_string(),
        })?;
    let covariance = select_survival_prediction_covariance(
        fit.beta_covariance(),
        fit.beta_covariance_corrected(),
        covariance_mode,
    )?;
    build_truncated_coefficient_draws(&fit, covariance)
        .map_err(|reason| SurvivalPredictError::PosteriorCovariance { reason })
}

pub(super) fn survival_truncated_law_posterior_moments(
    req: SurvivalPredictRequest<'_>,
    covariance_mode: SurvivalPredictionCovarianceMode,
    band_level: Option<f64>,
) -> Result<(SurvivalPredictResult, SurvivalPosteriorMoments), SurvivalPredictError> {
    let draws = truncated_survival_posterior_draws(req.model, covariance_mode)?.ok_or_else(|| {
        SurvivalPredictError::UnsupportedConfiguration {
            reason: format!(
                "the truncated-law survival posterior needs a location-scale or Royston-Parmar \
                 fit whose {} posterior carries an active inequality cone",
                covariance_mode.as_str()
            ),
        }
    })?;
    if require_saved_survival_likelihood_mode(req.model)? != SurvivalLikelihoodMode::LocationScale {
        return survival_replayed_truncated_law_posterior_moments(
            req,
            covariance_mode,
            &draws,
            band_level,
        );
    }
    let (result, moments) = predict_survival_surfaces(
        SurvivalPredictRequest {
            with_uncertainty: false,
            estimand: SurvivalPredictEstimand::Plugin,
            ..req
        },
        covariance_mode,
        Some(SurvivalSurfacePosterior::TruncatedLaw(&draws, band_level)),
    )?;
    let moments = moments.ok_or_else(|| {
        "internal error: the truncated-law survival pass returned no posterior moments".to_string()
    })?;
    refuse_decreasing_survival(&result)?;
    Ok((result, moments))
}

/// [`SurvivalPosteriorIntegration::TruncatedLaw`] for a fit whose plug-in
/// survival law is replayed whole at each coefficient vector rather than from
/// a location-scale batch — the Royston-Parmar fit, whose monotone I-spline
/// baseline is the cone `β_j ≥ 0` (gam#3575).
///
/// Every node of the law's joint rule is a feasible coefficient vector, so its
/// baseline is monotone and [`predict_survival_coefficient_law`] evaluates it
/// as it evaluates the fit's own coefficients. The cells are the plug-in
/// surfaces flattened row-major, `k = row·n_times + time`, followed by one
/// linear-predictor cell per row, and are certified exactly as the
/// location-scale surfaces are ([`truncated_survival_surface_moments`]).
fn survival_replayed_truncated_law_posterior_moments(
    req: SurvivalPredictRequest<'_>,
    covariance_mode: SurvivalPredictionCovarianceMode,
    draws: &TruncatedCoefficientDraws,
    band_level: Option<f64>,
) -> Result<(SurvivalPredictResult, SurvivalPosteriorMoments), SurvivalPredictError> {
    fn plugin_request<'a: 'b, 'b>(
        req: &SurvivalPredictRequest<'a>,
        model: &'b SavedModel,
    ) -> SurvivalPredictRequest<'b> {
        SurvivalPredictRequest {
            model,
            data: req.data,
            col_map: req.col_map,
            training_headers: req.training_headers,
            primary_offset: req.primary_offset,
            noise_offset: req.noise_offset,
            time_grid: req.time_grid,
            with_uncertainty: false,
            estimand: SurvivalPredictEstimand::Plugin,
        }
    }
    let fit = fit_result_from_saved_model_for_prediction(req.model)?;
    let result = predict_survival(plugin_request(&req, req.model), covariance_mode)?;
    let (n_rows, n_times) = result.survival.dim();
    let surface_width = n_rows * n_times;
    let times = result.times.clone();
    let likelihood_mode = result.likelihood_mode;
    let node_cells = |node_fit: &UnifiedFitResult| -> Result<SurvivalNodeCells, String> {
        let draw_model = saved_model_with_survival_coefficients(req.model, &node_fit.beta)?;
        let draw =
            predict_survival_coefficient_law(plugin_request(&req, &draw_model), covariance_mode)?;
        if draw.survival.dim() != (n_rows, n_times)
            || draw.hazard.dim() != (n_rows, n_times)
            || draw.cumulative_hazard.dim() != (n_rows, n_times)
            || draw.linear_predictor.len() != n_rows
            || draw.times != times
            || draw.likelihood_mode != likelihood_mode
        {
            return Err(
                "truncated posterior survival node changed the prediction schema".to_string(),
            );
        }
        let mut eta = Array1::<f64>::zeros(surface_width + n_rows);
        let mut log_survival = Array1::<f64>::zeros(surface_width + n_rows);
        let mut hazard = Array1::<f64>::zeros(surface_width + n_rows);
        for row in 0..n_rows {
            let linear_predictor = draw.linear_predictor[row];
            eta[surface_width + row] = linear_predictor;
            for time in 0..n_times {
                let k = row * n_times + time;
                eta[k] = linear_predictor;
                log_survival[k] = -draw.cumulative_hazard[[row, time]];
                hazard[k] = draw.hazard[[row, time]];
            }
        }
        Ok(SurvivalNodeCells {
            eta,
            log_survival,
            hazard,
        })
    };
    let surface_cells: Vec<(usize, usize, usize)> = (0..n_rows)
        .flat_map(|row| (0..n_times).map(move |time| (row, time, row * n_times + time)))
        .collect();
    let eta_cells: Vec<usize> = (0..n_rows).map(|row| surface_width + row).collect();
    let mut moments = truncated_survival_surface_moments(
        draws,
        &fit,
        &node_cells,
        &surface_cells,
        &eta_cells,
        n_times,
    )?;
    if let Some(level) = band_level {
        moments.survival_band = Some(replayed_truncated_surface_band(
            &req,
            covariance_mode,
            draws,
            &fit,
            &result,
            level,
        )?);
    }
    // The plug-in survival is published beside the posterior mean
    // (`survival_plugin`), so its curve is held to the same domain as when it
    // is published alone.
    refuse_decreasing_survival(&result)?;
    Ok((result, moments))
}

/// One coefficient vector's survival cells, in the caller's flattened
/// `(row, time)` layout: the linear predictor, `log S`, and the hazard, signed
/// where the node's survival rises (the location-scale index's rate, the
/// Royston-Parmar `dH/dt`).
pub(super) struct SurvivalNodeCells {
    pub(super) eta: Array1<f64>,
    pub(super) log_survival: Array1<f64>,
    pub(super) hazard: Array1<f64>,
}

impl SurvivalNodeCells {
    /// `(S, f)` at flattened cell `k`, `f = S·h` the node's event density.
    fn survival_and_density(&self, k: usize) -> Result<(f64, f64), String> {
        let log_survival = self.log_survival[k];
        let survival = log_survival.exp();
        let density = conditional_event_density(survival, -log_survival, self.hazard[k])
            .map_err(String::from)?;
        Ok((survival, density))
    }
}

/// The posterior moments of the survival surfaces over the cone-truncated
/// coefficient law — location-scale (gam#3038) and Royston-Parmar (gam#3575) —
/// certified per cell.
///
/// Every node of the law's joint rule is a feasible coefficient vector, and the
/// surfaces are replayed at it through `node_cells`. Each replicate lattice keeps
/// its own sums; a cell is retired once the replicate standard error of its
/// moments, less the integrand's rounding, is within the law's certified
/// relative accuracy — survival as a fraction of `sqrt(E[S](1 − E[S]))` (the
/// largest standard deviation a variable in `[0, 1]` with that mean can have),
/// the event density as a fraction of the larger of `|E[f]|` and its posterior
/// standard deviation (the published hazard is `E[f]/E[S]`, a ratio, so the
/// density is resolved relative to itself; where the posterior spread of `f`
/// exceeds its mean — deep in a tail, where the survival certificate already
/// resolves `E[S]` only to a fraction of `sqrt(E[S])` — resolving it to a
/// fraction of that spread keeps the Monte Carlo error a fraction `tol` of the
/// posterior uncertainty the surface carries), the linear predictor as a
/// fraction of its posterior standard deviation. Past the rule's maximum node count the moments are
/// refused, naming the worst cell, never reported.
///
/// `surface_cells` lists `(row, time column, flattened cell)` for every
/// published cell after the origin; origin cells are `S = 1`, `f = h = 0`
/// exactly. `eta_cells` is each row's linear-predictor cell.
pub(super) fn truncated_survival_surface_moments(
    draws: &TruncatedCoefficientDraws,
    fit: &UnifiedFitResult,
    node_cells: &(dyn Fn(&UnifiedFitResult) -> Result<SurvivalNodeCells, String> + Sync),
    surface_cells: &[(usize, usize, usize)],
    eta_cells: &[usize],
    t_cols: usize,
) -> Result<SurvivalPosteriorMoments, String> {
    use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

    let rule = draws.rule();
    let replicates = rule.replicates();
    if replicates < 2 {
        return Err(format!(
            "survival truncated posterior surfaces need at least two replicate \
             lattices to certify on; the joint rule carries {replicates}"
        ));
    }
    let tolerance = rule.relative_tolerance();
    // The sums run about the plug-in cells so their rounding is relative to the
    // posterior spread, not to the values (see `ResponseMomentAccumulator`).
    let reference = node_cells(fit)?;
    let total_cells = reference.eta.len();
    let reference_surface = surface_cells
        .iter()
        .map(|&(_, _, k)| reference.survival_and_density(k))
        .collect::<Result<Vec<_>, String>>()?;
    let mut accumulators: Vec<SurfaceMomentAccumulator> =
        std::iter::repeat_with(|| SurfaceMomentAccumulator::new(total_cells))
            .take(replicates)
            .collect();
    let n_rows = eta_cells.len();
    let mut moments = SurvivalPosteriorMoments::zeros(n_rows, t_cols);
    moments.survival.fill(PosteriorMoment::point(1.0));
    let mut active_surface: Vec<usize> = (0..surface_cells.len()).collect();
    let mut active_eta: Vec<usize> = (0..n_rows).collect();
    let mut evaluated = 0usize;
    while !(active_surface.is_empty() && active_eta.is_empty()) {
        let target = if evaluated == 0 {
            rule.initial_points()
        } else {
            2 * evaluated
        };
        let cells = SurfaceMomentCells {
            surface_cells,
            eta_cells,
            reference: &reference,
            reference_surface: &reference_surface,
            active_surface: &active_surface,
            active_eta: &active_eta,
        };
        accumulators
            .as_mut_slice()
            .into_par_iter()
            .enumerate()
            .try_for_each(|(replicate, accumulator)| {
                let mut node_fit = fit.clone();
                rule.visit_nodes(
                    replicate,
                    evaluated,
                    target,
                    |log_weight, normal_coordinates, tangent| {
                        let coefficients = draws.coefficients(normal_coordinates, tangent)?;
                        assign_survival_fit_coefficients(&mut node_fit, &coefficients)
                            .map_err(String::from)?;
                        accumulator.push(&cells, log_weight, &node_cells(&node_fit)?)
                    },
                )
            })?;
        evaluated = target;
        let pooled = PooledSurfaceMoments::new(&accumulators)?;
        let mut worst: Option<(String, f64)> = None;
        let mut note_uncertified = |label: String, error: f64| {
            if worst.as_ref().is_none_or(|(_, current)| error > *current) {
                worst = Some((label, error));
            }
        };
        let mut uncertified_surface = Vec::with_capacity(active_surface.len());
        for &cell in &active_surface {
            let (row, time, k) = surface_cells[cell];
            let (reference_survival, reference_density) = reference_surface[cell];
            let survival = pooled.moments(|a| a.survival[k], reference_survival);
            let density = pooled.moments(|a| a.density[k], reference_density);
            let hazard = pooled.mean(|a| a.hazard[k]);
            let first = survival.mean.clamp(0.0, 1.0);
            // `S = exp(log S)` resolves a probability to `f64::EPSILON` absolute
            // and `f` to one part in `f64::EPSILON` of itself: spread within that
            // is the integrand's rounding, which no node count removes.
            let survival_error = certified_fraction(
                survival.mean_spread.max(survival.sd_spread) - f64::EPSILON,
                (first * (1.0 - first)).sqrt(),
            );
            let density_error = certified_fraction(
                density.mean_spread
                    - f64::EPSILON * reference_density.abs().max(density.mean.abs()),
                density.mean.abs().max(density.variance.sqrt()),
            );
            let error = survival_error.max(density_error);
            if error <= tolerance {
                moments.survival[[row, time]] =
                    PosteriorMoment::from_mean_variance(first, survival.variance);
                moments.density_mean[[row, time]] = density.mean;
                moments.hazard_mean[[row, time]] = hazard;
            } else {
                uncertified_surface.push(cell);
                note_uncertified(format!("row {row}, time column {time}"), error);
            }
        }
        let mut uncertified_eta = Vec::with_capacity(active_eta.len());
        for &row in &active_eta {
            let k = eta_cells[row];
            let reference_eta = reference.eta[k];
            let eta = pooled.moments(|a| a.eta[k], reference_eta);
            let error = certified_fraction(
                eta.mean_spread.max(eta.sd_spread)
                    - f64::EPSILON * reference_eta.abs().max(eta.mean.abs()),
                eta.variance.sqrt(),
            );
            if error <= tolerance {
                moments.eta[row] = PosteriorMoment::from_mean_variance(eta.mean, eta.variance);
            } else {
                uncertified_eta.push(row);
                note_uncertified(format!("row {row}'s linear predictor"), error);
            }
        }
        active_surface = uncertified_surface;
        active_eta = uncertified_eta;
        let nodes = evaluated * replicates;
        if let Some((cell, error)) = worst
            && nodes >= rule.maximum_points()
        {
            return Err(format!(
                "survival truncated posterior surfaces did not certify: after \
                 {nodes} joint cubature nodes over {replicates} replicate lattices, the replicate \
                 standard error at {cell}, less the integrand's rounding, is {error:.3e} of its \
                 scale, above the law's certified relative accuracy {tolerance:.1e}"
            ));
        }
    }
    Ok(moments)
}

/// `excess / scale`, with an excess within rounding certified outright and a
/// positive excess on a zero scale never certified.
fn certified_fraction(excess: f64, scale: f64) -> f64 {
    if excess <= 0.0 {
        0.0
    } else if scale > 0.0 {
        excess / scale
    } else {
        f64::INFINITY
    }
}

/// What one node's accumulation reads, shared by every replicate.
struct SurfaceMomentCells<'a> {
    surface_cells: &'a [(usize, usize, usize)],
    eta_cells: &'a [usize],
    reference: &'a SurvivalNodeCells,
    reference_surface: &'a [(f64, f64)],
    active_surface: &'a [usize],
    active_eta: &'a [usize],
}

/// One replicate lattice's weighted moments, on one log scale, over the
/// flattened cells: centred moments of the deviations of `S`, `f` and `η` from
/// the plug-in cell, and the raw hazard sum, which only decides between a zero
/// and an infinite published hazard where `E[S] = 0`.
struct SurfaceMomentAccumulator {
    log_scale: f64,
    weight_sum: f64,
    survival: Array1<PosteriorMoment>,
    density: Array1<PosteriorMoment>,
    hazard: Array1<f64>,
    eta: Array1<PosteriorMoment>,
}

impl SurfaceMomentAccumulator {
    fn new(cells: usize) -> Self {
        Self {
            log_scale: f64::NEG_INFINITY,
            weight_sum: 0.0,
            survival: Array1::from_elem(cells, PosteriorMoment::EMPTY),
            density: Array1::from_elem(cells, PosteriorMoment::EMPTY),
            hazard: Array1::zeros(cells),
            eta: Array1::from_elem(cells, PosteriorMoment::EMPTY),
        }
    }

    fn push(
        &mut self,
        cells: &SurfaceMomentCells<'_>,
        log_weight: f64,
        node: &SurvivalNodeCells,
    ) -> Result<(), String> {
        if log_weight > self.log_scale {
            let rescale = (self.log_scale - log_weight).exp();
            self.weight_sum *= rescale;
            self.hazard *= rescale;
            for moments in [&mut self.survival, &mut self.density, &mut self.eta] {
                moments.map_inplace(|moment| *moment = moment.scaled(rescale));
            }
            self.log_scale = log_weight;
        }
        let weight = (log_weight - self.log_scale).exp();
        // A node whose weight underflowed on this scale contributes nothing, and
        // must not turn an infinite hazard into `0·∞`.
        if weight == 0.0 {
            return Ok(());
        }
        self.weight_sum += weight;
        for &cell in cells.active_surface {
            let (_, _, k) = cells.surface_cells[cell];
            let (reference_survival, reference_density) = cells.reference_surface[cell];
            let (survival, density) = node.survival_and_density(k)?;
            self.survival[k].merge(weight, PosteriorMoment::point(survival - reference_survival));
            self.density[k].merge(weight, PosteriorMoment::point(density - reference_density));
            self.hazard[k] += weight * node.hazard[k];
        }
        for &row in cells.active_eta {
            let k = cells.eta_cells[row];
            self.eta[k].merge(weight, PosteriorMoment::point(node.eta[k] - cells.reference.eta[k]));
        }
        Ok(())
    }
}

/// The replicate lattices pooled on the heaviest replicate's log scale.
struct PooledSurfaceMoments<'a> {
    accumulators: &'a [SurfaceMomentAccumulator],
    pooling_scales: Vec<f64>,
    pooled_weight: f64,
}

/// One quantity's pooled mean and variance, and the replicate standard errors
/// of its mean and of its standard deviation.
struct CellMoments {
    mean: f64,
    variance: f64,
    mean_spread: f64,
    sd_spread: f64,
}

impl<'a> PooledSurfaceMoments<'a> {
    fn new(accumulators: &'a [SurfaceMomentAccumulator]) -> Result<Self, String> {
        if let Some(accumulator) = accumulators
            .iter()
            .find(|accumulator| !(accumulator.weight_sum.is_finite() && accumulator.weight_sum > 0.0))
        {
            return Err(format!(
                "survival truncated posterior surfaces: a replicate lattice \
                 accumulated no finite node weight (weight sum {})",
                accumulator.weight_sum
            ));
        }
        let top = accumulators
            .iter()
            .map(|accumulator| accumulator.log_scale)
            .fold(f64::NEG_INFINITY, f64::max);
        let pooling_scales: Vec<f64> = accumulators
            .iter()
            .map(|accumulator| (accumulator.log_scale - top).exp())
            .collect();
        let pooled_weight = accumulators
            .iter()
            .zip(&pooling_scales)
            .map(|(accumulator, scale)| scale * accumulator.weight_sum)
            .sum();
        Ok(Self {
            accumulators,
            pooling_scales,
            pooled_weight,
        })
    }

    /// The pooled weighted mean of one raw sum.
    fn mean(&self, sum: impl Fn(&SurfaceMomentAccumulator) -> f64) -> f64 {
        self.accumulators
            .iter()
            .zip(&self.pooling_scales)
            .filter(|(_, scale)| **scale > 0.0)
            .map(|(accumulator, scale)| scale * sum(accumulator))
            .sum::<f64>()
            / self.pooled_weight
    }

    /// The moments of a quantity whose centred moments of deviations from
    /// `reference` `deviations` reads, the replicates merged on the pooling
    /// scales.
    fn moments(
        &self,
        deviations: impl Fn(&SurfaceMomentAccumulator) -> PosteriorMoment,
        reference: f64,
    ) -> CellMoments {
        let mut means = Vec::with_capacity(self.accumulators.len());
        let mut standard_deviations = Vec::with_capacity(self.accumulators.len());
        let mut pooled = PosteriorMoment::EMPTY;
        for (accumulator, &scale) in self.accumulators.iter().zip(&self.pooling_scales) {
            let replicate = deviations(accumulator);
            means.push(replicate.mean());
            standard_deviations.push(replicate.variance().sqrt());
            pooled.merge(scale, replicate);
        }
        CellMoments {
            mean: reference + pooled.mean(),
            variance: pooled.variance(),
            mean_spread: replicate_standard_error(&means),
            sd_spread: replicate_standard_error(&standard_deviations),
        }
    }
}

/// The surface moments of a fixed set of coefficient nodes with weights (the
/// sigma-point rule's), each replayed through `node_cells` on the designs the
/// caller assembled once: `E[S]`, `E[f]`, `E[h]` on every surface cell and
/// `E[η]` per row, exactly the sums [`super::survival_sigma_point_posterior_moments`]
/// forms from whole re-predictions. A cell at or before the time origin is
/// survival one with no density, as the plug-in surfaces have it.
pub(super) fn sigma_node_surface_moments(
    nodes: &[(Array1<f64>, f64)],
    fit: &UnifiedFitResult,
    node_cells: &(dyn Fn(&UnifiedFitResult) -> Result<SurvivalNodeCells, String> + Sync),
    surface_cells: &[(usize, usize, usize)],
    eta_cells: &[usize],
    t_cols: usize,
) -> Result<SurvivalPosteriorMoments, String> {
    let n_rows = eta_cells.len();
    let mut moments = SurvivalPosteriorMoments::zeros(n_rows, t_cols);
    let mut on_surface = ndarray::Array2::<bool>::from_elem((n_rows, t_cols), false);
    for &(i, j, _) in surface_cells {
        on_surface[[i, j]] = true;
    }
    for (coefficients, weight) in nodes {
        let mut node_fit = fit.clone();
        assign_survival_fit_coefficients(&mut node_fit, coefficients).map_err(String::from)?;
        let cells = node_cells(&node_fit)?;
        for &(i, j, k) in surface_cells {
            let (survival, density) = cells.survival_and_density(k)?;
            moments.survival[[i, j]].merge(*weight, PosteriorMoment::point(survival));
            moments.density_mean[[i, j]] += weight * density;
            moments.hazard_mean[[i, j]] += weight * cells.hazard[k];
        }
        for ((i, j), on) in on_surface.indexed_iter() {
            if !on {
                moments.survival[[i, j]].merge(*weight, PosteriorMoment::point(1.0));
            }
        }
        for (row, &k) in eta_cells.iter().enumerate() {
            moments.eta[row].merge(*weight, PosteriorMoment::point(cells.eta[k]));
        }
    }
    Ok(moments)
}
