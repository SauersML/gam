//! History-conditioned forecasting, the population tier of the same
//! forecast, and the predictive probability integral transform, all
//! expectations under the latent state.
//!
//! A forecast filters the subject's own history into its latent state and
//! then integrates the killed process forward: the survival to a horizon is
//! `E[exp(−∫ Λ_T(t) dt)]` with `Λ_T` the total intensity of the terminal
//! marks, and the expected count of a mark by a horizon is the chronological
//! integral of the sub-density `m_d(t) = E[S(t) λ_d(t)]`, where `S(t)` is the
//! survival of the killed process up to `t` along the same latent path.
//! For a terminal mark that integral is its cumulative incidence; for a
//! once-only mark it is the probability of a first occurrence before
//! termination, with the mark's own hazard added to the killing; for a
//! recurrent mark it is the expected number of events before termination.
//!
//! The chronology is real: the survival at an interior quadrature time is a
//! Gauss-Legendre integral over the elapsed time up to that point, computed
//! by its own filter chain from the state at the start of the mesh cell, so
//! every reported probability is a proper quadrature of a well-defined
//! integral. The terminal marks share each cell's survival decrement in
//! proportion to their sub-densities, so survival and terminal incidence are
//! one evolution and `Σ_{d terminal} F_d(h) = 1 − S(h)` holds to roundoff.
//!
//! A window's accuracy is its own, not the fit's. A training certificate
//! says the fitted coefficients are stationary under refinement; it says
//! nothing about a different integral over a future window. So the window's
//! mesh is refined until the time error of what it returns no longer
//! dominates the other measured components of its error or its roundoff, and
//! that total error is returned with it. Under a latent state the other
//! component is the Gauss-Hermite quadrature: the same window on the next
//! rung `2G − 1`, which filters the history and evolves its state under its
//! own rule, so the gap between the two includes every error the state has
//! accumulated, not only a cell's own. The horizons are output times, never
//! mesh breakpoints, so requesting more of them cannot move a prediction (see
//! `integrator`).
//!
//! The same window run from the stationary prior instead of a filtered
//! state ([`population_forecast`]) gives the lower tiers of the information
//! hierarchy: with population covariate values it is the population risk;
//! with a subject's own covariates (a risk score) it is what the model says
//! before any history is observed; the history-conditioned forecast updates
//! it. No weight between the tiers is chosen by hand — each is the same
//! probability model conditioned on more.
//!
//! The predictive PIT of a spell — the follow-up from one event (or the
//! entry) to the next event or to the exit — is `1 − P(no event of any mark
//! in the spell | history before it)`. It conditions on nothing recorded after
//! the spell, so appending later records cannot change it: a dynamic factor's
//! filter yields it as the product of the normalisers of the zero-count nodes
//! in the spell, and a static factor as a ratio of prefix integrals, each on
//! its own grid (`super::static_state`). Under the model, the
//! PIT of a spell that ends in an event is a uniform, and the sequence over
//! a subject's spells is a Rosenblatt transform of its event times:
//! independent uniforms across events and across subjects (the
//! time-rescaling theorem). A spell that ends at the exit without an event
//! is a censored draw of that uniform: all the model says is that its PIT
//! exceeds the value emitted. Dropping those spells and comparing the event
//! PITs alone to the uniform law is wrong under censoring — with a constant
//! hazard `λ` observed to `c`, the event PITs are uniform on `[0, 1 − e^{−λc}]`,
//! and their distance from the uniform law on `[0, 1]` tends to `e^{−λc}`
//! under a perfectly specified model. [`pit_uniform_distance`] therefore
//! estimates the PIT distribution by Kaplan–Meier over event and censored
//! spells alike and measures its distance from the uniform law, which
//! reduces to the ordinary Kolmogorov–Smirnov distance when nothing is
//! censored. The predictive mark probabilities at each event complete the
//! diagnostic for marked processes.
//!
//! One scope limit under the risk-set centring: [`population_forecast`] opens
//! its window at the *stationary prior*, which is the law of the latent state
//! over the cohort as it started, not over those still at risk at the window's
//! start. The two agree at the first age the reference population is run from
//! and separate as the risk set is selected, so a population tier taken late
//! in follow-up reads above the incidence its own baseline names. A
//! history-conditioned forecast is unaffected: it opens at the state that
//! history implies. Opening the population tier at the reference population's
//! own risk-set law is the fix, and it needs that law carried out of the
//! reference filter, which is not yet done.
//!
//! A forecast can be made for any history, not only a training subject's:
//! [`forecast_history`] takes a history with its own covariate rows, and
//! [`SubjectHistory::prefix`] cuts a history at an assessment time, so that
//! a forecast made at a cutoff sees exactly what was known then and cannot
//! change when later records are appended.

use super::chain::{GaussHermite, Grid, product_grid_size};
use super::cohort::{
    CohortNodes, CovariateSegment, EventHistoryCohort, EventHistoryError, MarkKind, SubjectHistory,
    SubjectNodes, cell_rule, expand_nodes, mesh_cells,
};
use super::family::EventHistoryFit;
use super::marginal::{
    ForwardPass, SubjectInputs, expected_intensities, forward_filter, latent_state_moments, spells,
};
use gam_terms::smooth::build_term_collection_design;
use ndarray::{Array2, ArrayView2};

pub(crate) mod integrator;

use integrator::{CellSums, Companion, KilledProcess, KilledRun, integrate_window};

/// One piece of a forecast window during which the covariates are
/// constant: from `start` until the next segment's start (or the last
/// horizon). `covariates` holds one value per covariate column, with
/// categorical covariates as their level codes.
#[derive(Clone, Debug, PartialEq)]
pub struct FutureSegment {
    pub start: f64,
    pub covariates: Vec<f64>,
}

/// A forecast request for one subject.
pub struct ForecastRequest<'a> {
    /// The subject's observed history (its covariate rows index the cohort).
    pub history: &'a SubjectHistory,
    /// Which reference stratum this person belongs to, when the fit's
    /// baselines are the risk sets' rates. Ignored otherwise.
    pub stratum: usize,
    /// Absolute horizon times, strictly increasing and after the exit time.
    pub horizons: &'a [f64],
    /// The covariate path over the forecast window. Empty holds the row in
    /// force at exit; otherwise a first segment starting after exit is
    /// preceded by that row.
    pub future: &'a [FutureSegment],
}

/// A forecast for a subject with no observed history: the latent state
/// starts at its stationary prior at `start`, so this is the population
/// tier when the covariates are population values and the covariate-only
/// tier (a new subject with a risk score) otherwise.
pub struct PopulationForecastRequest<'a> {
    /// Time the forecast window opens.
    pub start: f64,
    /// Which reference stratum the window's population belongs to, when the
    /// fit's baselines are the risk sets' rates. Ignored otherwise.
    pub stratum: usize,
    /// Absolute horizon times, strictly increasing and after `start`.
    pub horizons: &'a [f64],
    /// The covariate path over the window; the first segment must start at
    /// or before `start`.
    pub future: &'a [FutureSegment],
}

/// A forecast request for a history that is not a training subject's: the
/// history's covariate segments index `covariates`, the subject's own rows
/// in the cohort's columns (categorical covariates as level codes).
pub struct HistoryForecastRequest<'a> {
    pub history: &'a SubjectHistory,
    pub covariates: ArrayView2<'a, f64>,
    /// Which reference stratum this person belongs to, when the fit's
    /// baselines are the risk sets' rates. Ignored otherwise.
    pub stratum: usize,
    /// Absolute horizon times, strictly increasing and after the exit time.
    pub horizons: &'a [f64],
    /// The covariate path over the forecast window, as in [`ForecastRequest`].
    pub future: &'a [FutureSegment],
}

/// A forecast: per horizon, the probability that no terminal event has
/// fired, and the expected count of every mark (its cumulative incidence
/// when terminal, its first-occurrence probability when once-only), each
/// with the numerical error its integration checked.
#[derive(Clone, Debug)]
pub struct Forecast {
    pub horizons: Vec<f64>,
    pub survival: Vec<f64>,
    pub expected_counts: Array2<f64>,
    /// The refinement discrepancies the window's integration accepted,
    /// accumulated to each horizon: the checked numerical error of
    /// `survival`.
    pub survival_error: Vec<f64>,
    /// The same for every entry of `expected_counts`.
    pub expected_count_errors: Array2<f64>,
}

/// The predictive PIT of one spell of a subject's follow-up: from the
/// previous event (or the entry) to the next event, or to the exit when no
/// further event was observed.
#[derive(Clone, Debug)]
pub struct SpellPit {
    /// When the spell ended: the event time, or the exit of a censored spell.
    pub time: f64,
    /// Whether the spell ended with an event. A censored spell's `pit` is a
    /// lower bound of the uniform the model assigns to it, not a draw.
    pub observed: bool,
    /// `1 − P(no event of any mark in the spell | history)`.
    pub pit: f64,
    /// The marks that fired at the spell's end, one entry per event (empty
    /// for a censored spell).
    pub marks: Vec<usize>,
    /// The predictive probability of each mark given that an event happened
    /// at the spell's end.
    pub mark_probabilities: Vec<f64>,
}

/// `log M_d(t)` at a set of node times for one stratum, laid out
/// `n * marks + d` as the filter reads it. Empty when the fit's baselines are
/// centred on the stationary prior, which is what tells the filter to use the
/// prior's constant shift instead.
///
/// A forecast from a risk-set centred fit has to divide by the same
/// normaliser the fit did, at the times the forecast asks about. Leaving it
/// out would evaluate a different model from the one that was fitted — one
/// whose baselines are rates over the cohort as it started — and the
/// difference is exactly the selection the centring exists to account for.
fn forecast_normaliser(fit: &EventHistoryFit, stratum: usize, times: &[f64]) -> Result<Option<Vec<f64>>, EventHistoryError> {
    let mut out = Vec::with_capacity(times.len() * fit.marks());
    for &t in times {
        out.extend(fit.risk_set_normaliser_at(stratum, t)?);
    }
    Ok((!out.is_empty()).then_some(out))
}

fn single_subject_nodes(
    fit: &EventHistoryFit,
    cohort: &EventHistoryCohort,
    table: ArrayView2<'_, f64>,
    history: &SubjectHistory,
) -> Result<CohortNodes, EventHistoryError> {
    let mut one = EventHistoryCohort {
        mark_names: cohort.mark_names.clone(),
        mark_kinds: cohort.mark_kinds.clone(),
        covariate_names: cohort.covariate_names.clone(),
        covariate_levels: cohort.covariate_levels.clone(),
        covariates: table.to_owned(),
        subjects: vec![history.clone()],
    };
    one.validate()?;
    expand_nodes(&one, fit.quadrature_order, fit.mesh_refinement)
}

/// Population node log-intensities `η⁰` (design × coefficients + offset)
/// for every mark on a row matrix (covariate columns then time), index
/// `row * marks + d`.
fn node_eta0(
    fit: &EventHistoryFit,
    rows: ArrayView2<'_, f64>,
) -> Result<Vec<f64>, EventHistoryError> {
    let marks = fit.marks();
    let total = rows.nrows();
    let mut eta0 = vec![0.0; total * marks];
    for d in 0..marks {
        let design = build_term_collection_design(rows, &fit.frozen_specs[d]).map_err(|error| {
            EventHistoryError::Fit {
                reason: format!("prediction design for mark {d}: {error}"),
            }
        })?;
        let beta = fit.mark_coefficients(d);
        if design.design.ncols() != beta.len() {
            return Err(EventHistoryError::Fit {
                reason: format!(
                    "prediction design for mark {d} has {} columns, the fit has {} coefficients",
                    design.design.ncols(),
                    beta.len()
                ),
            });
        }
        let dense = design
            .design
            .try_to_dense_arc("event-history prediction design")
            .map_err(|error| EventHistoryError::Fit {
                reason: error.to_string(),
            })?;
        for row in 0..total {
            let mut value = design.affine_offset[row];
            for (j, x) in dense.row(row).iter().enumerate() {
                value += x * beta[j];
            }
            eta0[row * marks + d] = value;
        }
    }
    Ok(eta0)
}

/// The fitted baseline surfaces, evaluated directly on covariate rows with
/// time in the last column. This is distinct from averaging individual
/// forecasts under an entry-conditioned population.
pub fn baseline_log_rates(fit: &EventHistoryFit, rows: ArrayView2<'_, f64>) -> Result<Array2<f64>, EventHistoryError> {
    if rows.ncols() != fit.nodes.time_column + 1 || rows.iter().any(|x| !x.is_finite()) {
        return Err(EventHistoryError::InvalidInput {
            reason: "baseline rows need finite covariates followed by time".to_string(),
        });
    }
    let values = node_eta0(fit, rows)?;
    Array2::from_shape_vec((rows.nrows(), fit.marks()), values)
        .map_err(|error| EventHistoryError::InvalidInput { reason: error.to_string() })
}

/// The smoothed latent state of one subject: at every node of its history,
/// the posterior mean of the atoms and their posterior covariance given the
/// whole history. This is the quantity a discovery analysis wants — with its
/// covariance, so that a fitted path is propagated as the uncertain object
/// it is rather than read as observed.
#[derive(Clone, Debug)]
pub struct SmoothedLatentState {
    /// The node times, entry to exit.
    pub times: Vec<f64>,
    /// `E[z(t) | history]`, `nodes × atoms`.
    pub mean: Array2<f64>,
    /// `Cov[z(t) | history]`, one `atoms × atoms` matrix per node.
    pub covariance: Vec<Array2<f64>>,
}

/// The smoothed latent state of a subject over its observed history.
pub fn latent_state(
    fit: &EventHistoryFit,
    cohort: &EventHistoryCohort,
    history: &SubjectHistory,
    stratum: usize,
) -> Result<SmoothedLatentState, EventHistoryError> {
    let atoms = fit.rank();
    let (loadings, rates) = latent_parameters(fit);
    let nodes = single_subject_nodes(fit, cohort, cohort.covariates.view(), history)?;
    let eta0 = node_eta0(fit, nodes.node_data.view())?;
    let subject = &nodes.subjects[0];
    let normaliser = forecast_normaliser(fit, stratum, &subject.times)?;
    let moments = latent_state_moments(&SubjectInputs {
        nodes: subject,
        eta0: &eta0,
        loadings: &loadings,
        rates: &rates,
        time_scale: fit.time_scale,
        gh: fit.family.gauss_hermite(),
        continuation_gap: 0.0,
        designs: None,
        log_normaliser: normaliser.as_deref(),
    })?;
    let mut mean = Array2::<f64>::zeros((subject.len(), atoms));
    let mut covariance = Vec::with_capacity(subject.len());
    for (n, (node_mean, node_covariance)) in moments.iter().enumerate() {
        for k in 0..atoms {
            mean[[n, k]] = node_mean[k];
        }
        let mut matrix = Array2::<f64>::zeros((atoms, atoms));
        for k in 0..atoms {
            for j in 0..atoms {
                matrix[[k, j]] = node_covariance[k * atoms + j];
            }
        }
        covariance.push(matrix);
    }
    Ok(SmoothedLatentState {
        times: subject.times.clone(),
        mean,
        covariance,
    })
}

/// The loadings (index `d * atoms + k`) and the dimensionless rates `ν` of a
/// fit, as the filter takes them.
fn latent_parameters(fit: &EventHistoryFit) -> (Vec<f64>, Vec<f64>) {
    let marks = fit.marks();
    let atoms = fit.rank();
    let mut loadings = Vec::with_capacity(marks * atoms);
    for d in 0..marks {
        for k in 0..atoms {
            loadings.push(fit.loadings[[d, k]]);
        }
    }
    (loadings, fit.log_rates.iter().map(|r| r.exp()).collect())
}

/// A filtered latent state: the grid and density at a time.
#[derive(Clone)]
struct LatentState {
    grid: Grid<f64>,
    alpha: Vec<f64>,
    time: f64,
}

/// The observed history a forecast window opens after: its covariate rows
/// (which its segments index), the history, and the marks whose compensators
/// it carries.
struct Opening<'a> {
    table: ArrayView2<'a, f64>,
    history: SubjectHistory,
    compensated: Vec<bool>,
}

/// The filtered latent state after an opening history, under Gauss-Hermite
/// rule `gh`. Each rule a window integrates under filters the history anew,
/// placing its own grids, so the state it continues from is its own.
fn observed_state(
    fit: &EventHistoryFit,
    cohort: &EventHistoryCohort,
    opening: &Opening<'_>,
    stratum: usize,
    loadings: &[f64],
    rates: &[f64],
    gh: &GaussHermite,
) -> Result<LatentState, EventHistoryError> {
    let history = &opening.history;
    let observed = single_subject_nodes(fit, cohort, opening.table, history)?;
    chain_fits(gh.order, fit.rank(), observed.subjects[0].len())?;
    let eta0 = node_eta0(fit, observed.node_data.view())?;
    let normaliser = forecast_normaliser(fit, stratum, &observed.subjects[0].times)?;
    let mut pass = forward_filter(
        &SubjectInputs {
            nodes: &observed.subjects[0],
            eta0: &eta0,
            loadings,
            rates,
            time_scale: fit.time_scale,
            gh,
            continuation_gap: 0.0,
            designs: None,
            log_normaliser: normaliser.as_deref(),
        },
        None,
        &opening.compensated,
    )?;
    let last = observed.subjects[0].len() - 1;
    Ok(LatentState {
        grid: pass.grids.pop().expect("at least one node"),
        alpha: pass.alpha.pop().expect("at least one node"),
        time: observed.subjects[0].times[last],
    })
}

/// A chain of zero-count nodes for the future filter.
fn future_chain(times: &[f64], weights: &[f64], exposed: &[bool], marks: usize) -> SubjectNodes {
    let n = times.len();
    let mut exposures = Array2::<f64>::zeros((n, marks));
    for (i, &w) in weights.iter().enumerate() {
        for d in 0..marks {
            if exposed[d] {
                exposures[[i, d]] = w;
            }
        }
    }
    SubjectNodes {
        first_row: 0,
        times: times.to_vec(),
        gaps: times.windows(2).map(|w| w[1] - w[0]).collect(),
        weights: weights.to_vec(),
        exposures,
        counts: Array2::zeros((n, marks)),
        covariate_rows: vec![0; n],
    }
}

fn validate_horizons(horizons: &[f64], start: f64) -> Result<(), EventHistoryError> {
    if horizons.is_empty()
        || horizons.iter().any(|h| !h.is_finite())
        || horizons.windows(2).any(|w| !(w[1] > w[0]))
        || !(horizons[0] > start)
    {
        return Err(EventHistoryError::InvalidInput {
            reason: format!(
                "horizons must be finite, strictly increasing and later than the window's start {start}"
            ),
        });
    }
    Ok(())
}

fn validate_future(
    cohort: &EventHistoryCohort,
    future: &[FutureSegment],
    last_horizon: f64,
) -> Result<(), EventHistoryError> {
    let n_cov = cohort.covariates.ncols();
    for (i, segment) in future.iter().enumerate() {
        if !segment.start.is_finite() || segment.covariates.len() != n_cov {
            return Err(EventHistoryError::InvalidInput {
                reason: format!(
                    "future segment {i} needs a finite start and {n_cov} covariate values, got start {} and {} values",
                    segment.start,
                    segment.covariates.len()
                ),
            });
        }
        if i > 0 && !(segment.start > future[i - 1].start) {
            return Err(EventHistoryError::InvalidInput {
                reason: "future segments must have strictly increasing starts".to_string(),
            });
        }
        if segment.start >= last_horizon {
            return Err(EventHistoryError::InvalidInput {
                reason: format!(
                    "future segment {i} starts at {}, at or after the last horizon",
                    segment.start
                ),
            });
        }
        for (j, value) in segment.covariates.iter().enumerate() {
            if !value.is_finite() {
                return Err(EventHistoryError::InvalidInput {
                    reason: format!(
                        "future segment {i} has a non-finite covariate value at column {j}"
                    ),
                });
            }
            let levels = &cohort.covariate_levels[j];
            if !levels.is_empty()
                && (value.fract() != 0.0 || *value < 0.0 || *value >= levels.len() as f64)
            {
                return Err(EventHistoryError::InvalidInput {
                    reason: format!(
                        "future segment {i} has code {value} for categorical covariate {:?} with {} levels",
                        cohort.covariate_names[j],
                        levels.len()
                    ),
                });
            }
        }
    }
    Ok(())
}

/// One forecast window: where it opens, the covariate path over it, the
/// history whose filtered state it continues from (`None` for the stationary
/// prior), and which marks the subject is still at risk for.
struct Window<'a> {
    fit: &'a EventHistoryFit,
    cohort: &'a EventHistoryCohort,
    stratum: usize,
    opening: Option<Opening<'a>>,
    start: f64,
    horizons: &'a [f64],
    segments: Vec<CovariateSegment>,
    /// The cohort's covariate table extended by the window's own rows.
    table: Array2<f64>,
    at_risk: Vec<bool>,
    label: String,
}

/// Refuse a filter chain of `chain_nodes` nodes under Gauss-Hermite order
/// `order` that would not fit this machine's materialisation budget: every
/// node of a chain keeps its grid weights and its predicted and filtered
/// densities.
fn chain_fits(order: usize, atoms: usize, chain_nodes: usize) -> Result<(), EventHistoryError> {
    let points = product_grid_size(order, atoms)?;
    let bytes = 3.0 * chain_nodes as f64 * points as f64 * std::mem::size_of::<f64>() as f64;
    let budget = gam_runtime::resource::ResourcePolicy::default_library()
        .max_single_materialization_bytes as f64;
    if bytes > budget {
        return Err(EventHistoryError::NumericalFailure {
            reason: format!(
                "a forecast's filter chain of {chain_nodes} nodes at Gauss-Hermite order {order} over {atoms} atoms ({points} grid points) needs about {:.1} GiB, above this machine's {:.1} GiB materialisation budget",
                bytes / f64::from(1u32 << 30),
                budget / f64::from(1u32 << 30)
            ),
        });
    }
    Ok(())
}

/// The grid-filter engine of a forecast window under one Gauss-Hermite rule:
/// the fit, the covariate path over the window, the latent parameters, the
/// Gauss-Legendre rule per cell, and the window's killed runs. A latent window
/// runs two, on the fit's rule and on its next rung.
struct WindowIntegrand<'a> {
    fit: &'a EventHistoryFit,
    stratum: usize,
    pseudo: &'a SubjectHistory,
    table: &'a Array2<f64>,
    loadings: &'a [f64],
    rates: &'a [f64],
    gl_nodes: &'a [f64],
    gl_weights: &'a [f64],
    gh: &'a GaussHermite,
    /// Whether the filter interpolates its density onto a new grid at every
    /// node. A dynamic factor's does: `marginal::filter_step` carries the
    /// density through `chain::forward_operators`, which evaluates the rule's
    /// Lagrange basis at every target point (`gh.lagrange_basis(&raw)`). A
    /// static factor's grid is conditioned in place (`static_state::filter`),
    /// and a rank-zero window has no grid.
    interpolates: bool,
    runs: &'a [KilledRun],
}

impl WindowIntegrand<'_> {
    /// Every run's cell sums over `[left, right]` from `from` under rule `gh`.
    /// The survival at every outer node is its own filter chain from the
    /// cell's start, and every at-risk mark's sub-density
    /// `S(t) E[λ_d(t) | alive]` is summed on the outer rule.
    fn cell_under(
        &self,
        gh: &GaussHermite,
        left: f64,
        right: f64,
        from: &[Option<LatentState>],
    ) -> Result<Vec<CellSums<Option<LatentState>>>, EventHistoryError> {
        let fit = self.fit;
        let marks = fit.marks();
        let atoms = fit.rank();
        let q = self.gl_nodes.len();
        let outer: Vec<(f64, f64)> = cell_rule(left, right, self.gl_nodes, self.gl_weights).collect();
        let inner: Vec<Vec<(f64, f64)>> = outer
            .iter()
            .map(|&(t, _)| cell_rule(left, t, self.gl_nodes, self.gl_weights).collect())
            .collect();
        // Evaluation points: the outer nodes, then every outer node's own
        // chronological rule from the cell's start.
        let times: Vec<f64> = outer.iter().chain(inner.iter().flatten()).map(|&(t, _)| t).collect();
        let n_cov = self.table.ncols();
        let mut rows = Array2::<f64>::zeros((times.len(), n_cov + 1));
        for (i, &t) in times.iter().enumerate() {
            let row = self.pseudo.covariate_row_at(t, false);
            for j in 0..n_cov {
                rows[[i, j]] = self.table[[row, j]];
            }
            rows[[i, n_cov]] = t;
        }
        let eta0 = node_eta0(fit, rows.view())?;
        let outer_times: Vec<f64> = times[..q].to_vec();
        let outer_weights: Vec<f64> = outer.iter().map(|&(_, w)| w).collect();
        let mut sums = Vec::with_capacity(self.runs.len());
        for (run, state) in self.runs.iter().zip(from) {
            let state = state.as_ref();
            let filter = |times: &[f64], weights: &[f64], eta: &[f64]| -> Result<ForwardPass<f64>, EventHistoryError> {
                let nodes = future_chain(times, weights, &run.exposed, marks);
                let normaliser = forecast_normaliser(fit, self.stratum, times)?;
                forward_filter(
                    &SubjectInputs {
                        nodes: &nodes,
                        eta0: eta,
                        loadings: self.loadings,
                        rates: self.rates,
                        time_scale: fit.time_scale,
                        gh,
                        continuation_gap: state.map_or(0.0, |s| times[0] - s.time),
                        designs: None,
                        log_normaliser: normaliser.as_deref(),
                    },
                    state.map(|s| (&s.grid, s.alpha.as_slice())),
                    &run.exposed,
                )
            };
            let mut sub_densities = vec![0.0; marks];
            for j in 0..q {
                let first = q + j * q;
                let mut chain_times = times[first..first + q].to_vec();
                let mut chain_weights: Vec<f64> = inner[j].iter().map(|&(_, w)| w).collect();
                chain_times.push(outer_times[j]);
                chain_weights.push(0.0);
                let mut chain_eta = eta0[first * marks..(first + q) * marks].to_vec();
                chain_eta.extend_from_slice(&eta0[j * marks..(j + 1) * marks]);
                let pass = filter(&chain_times, &chain_weights, &chain_eta)?;
                let survival = pass.log_normalisers.iter().sum::<f64>().exp();
                let at_j = forecast_normaliser(fit, self.stratum, &outer_times[j..j + 1])?;
                let intensities = expected_intensities(
                    &pass.grids[q],
                    &pass.predicted[q],
                    &eta0[j * marks..(j + 1) * marks],
                    self.loadings,
                    at_j.as_deref(),
                    marks,
                    atoms,
                );
                for d in 0..marks {
                    if run.reported[d] || run.exposed[d] {
                        sub_densities[d] += outer_weights[j] * survival * intensities[d];
                    }
                }
            }
            // Advance the state across the cell along its own rule.
            let mut pass = filter(&outer_times, &outer_weights, &eta0[..q * marks])?;
            sums.push(CellSums {
                log_decrement: pass.log_normalisers.iter().sum(),
                sub_densities,
                state: Some(LatentState {
                    grid: pass.grids.pop().expect("cell has nodes"),
                    alpha: pass.alpha.pop().expect("cell has nodes"),
                    time: outer_times[q - 1],
                }),
            });
        }
        Ok(sums)
    }
}

impl KilledProcess for WindowIntegrand<'_> {
    type State = Option<LatentState>;

    fn runs(&self) -> &[KilledRun] {
        self.runs
    }

    fn cell(
        &self,
        left: f64,
        right: f64,
        from: &[Option<LatentState>],
    ) -> Result<Vec<CellSums<Option<LatentState>>>, EventHistoryError> {
        self.cell_under(self.gh, left, right, from)
    }

    /// `ε · Λ · nodes`, over the nodes a halved cell's chronology filters. A
    /// dynamic factor's filter carries its density onto a new grid at every
    /// node through the rule's Lagrange interpolant, which amplifies nodal
    /// roundoff by the rule's Lebesgue constant `Λ_G`: the charge the fit's own
    /// certificate reads for the same route. It is this rule's constant, the
    /// rule whose values the window returns. A companion's values enter only its
    /// measured gap, so the next rung's constant, which grows exponentially with
    /// the order, is never charged to them. A static factor's grid is
    /// conditioned in place and a rank-zero window has no grid: nothing is
    /// interpolated, and `Λ` is one.
    fn roundoff(&self) -> f64 {
        let amplification = if self.interpolates { self.gh.lebesgue_constant } else { 1.0 };
        f64::EPSILON * amplification * (2 * self.gl_nodes.len() + 1) as f64
    }
}

/// The killed-process integration of one forecast window, on the grid
/// filter under the fit's parameters. The window's level-0 breakpoints are
/// its start, the covariate path's changes and the last horizon; the mesh,
/// its acceptance and the horizons are `integrator::integrate_window`'s. The
/// fit's quadrature order serves only as the rule per cell, and its mesh
/// refinement not at all.
///
/// Under a latent state the window's companion is the same window on the
/// next rung `2G − 1` of the fit's Gauss-Hermite rule: its opening state is
/// the history filtered under that rung, which places its own grids, and it
/// evolves under that rung over the same mesh. The window returns the fit's
/// rule's values, whose latent error the gap to the companion measures on the
/// premise that the next rung is the more accurate of the two.
///
/// The companion's filter chains are materialised like the window's own. A
/// cell's chain, or the history the window opens after, that would not fit
/// this machine's materialisation budget under the rung refuses the forecast
/// (`chain_fits`): at rank two or more, a long history can be refused on the
/// companion's budget alone.
fn run_window(window: Window<'_>) -> Result<Forecast, EventHistoryError> {
    let Window {
        fit,
        cohort,
        stratum,
        opening,
        start,
        horizons,
        segments,
        table,
        at_risk,
        label,
    } = window;
    let marks = fit.marks();
    let kinds = &cohort.mark_kinds;
    let terminal: Vec<bool> = kinds.iter().map(|k| *k == MarkKind::Terminal).collect();
    let mut runs = vec![KilledRun {
        exposed: (0..marks).map(|d| terminal[d] && at_risk[d]).collect(),
        reported: (0..marks)
            .map(|d| at_risk[d] && kinds[d] != MarkKind::Once)
            .collect(),
    }];
    for d in 0..marks {
        if kinds[d] == MarkKind::Once && at_risk[d] {
            runs.push(KilledRun {
                exposed: (0..marks).map(|k| (terminal[k] || k == d) && at_risk[k]).collect(),
                reported: (0..marks).map(|k| k == d).collect(),
            });
        }
    }
    let (loadings, rates) = latent_parameters(fit);
    let (gl_nodes, gl_weights) = gam_math::special::gauss_legendre(fit.quadrature_order);
    let gh: &GaussHermite = fit.family.gauss_hermite();
    let rung = if fit.rank() > 0 {
        let order = 2 * gh.order - 1;
        chain_fits(order, fit.rank(), fit.quadrature_order + 1)?;
        Some(GaussHermite::new(order)?)
    } else {
        None
    };
    let pseudo = SubjectHistory {
        id: label,
        entry: start,
        exit: horizons[horizons.len() - 1],
        events: Vec::new(),
        segments,
    };
    let integrand = WindowIntegrand {
        fit,
        stratum,
        pseudo: &pseudo,
        table: &table,
        loadings: &loadings,
        rates: &rates,
        gl_nodes: &gl_nodes,
        gl_weights: &gl_weights,
        gh,
        interpolates: fit.rank() > 0 && !crate::static_state::is_static(&rates),
        runs: &runs,
    };
    // Each rule's opening state is the history filtered under that rule, or
    // the stationary prior.
    let opened = |rule: &GaussHermite| -> Result<Vec<Option<LatentState>>, EventHistoryError> {
        let state = opening
            .as_ref()
            .map(|history| observed_state(fit, cohort, history, stratum, &loadings, &rates, rule))
            .transpose()?;
        Ok(vec![state; runs.len()])
    };
    let companion_integrand = rung.as_ref().map(|rule| WindowIntegrand { gh: rule, ..integrand });
    let companion = match &companion_integrand {
        Some(beside) => Some(Companion {
            process: beside,
            opening: opened(beside.gh)?,
        }),
        None => None,
    };
    let mut breakpoints = vec![start];
    breakpoints.extend(
        mesh_cells(&pseudo, false, 0)
            .iter()
            .map(|&(_, right)| right),
    );
    let reached = integrate_window(&integrand, opened(gh)?, companion, &breakpoints, horizons)?;
    let n_h = horizons.len();
    let mut survival = vec![0.0; n_h];
    let mut survival_error = vec![0.0; n_h];
    let mut expected = Array2::<f64>::zeros((n_h, marks));
    let mut expected_errors = Array2::<f64>::zeros((n_h, marks));
    for (i, positions) in reached.iter().enumerate() {
        survival[i] = positions[0].log_survival.exp();
        survival_error[i] = positions[0].survival_error;
        for (run, p) in integrand.runs.iter().zip(positions) {
            for d in 0..marks {
                if run.reported[d] {
                    expected[[i, d]] = p.counts[d];
                    expected_errors[[i, d]] = p.count_errors[d];
                }
            }
        }
    }
    Ok(Forecast {
        horizons: horizons.to_vec(),
        survival,
        expected_counts: expected,
        survival_error,
        expected_count_errors: expected_errors,
    })
}

/// The covariate table `base` extended by the future segments' rows, and
/// the segments as rows of that table starting at `window_start` (with
/// `initial_row` in force before the first future segment, if any).
fn future_table(
    base: ArrayView2<'_, f64>,
    future: &[FutureSegment],
    window_start: f64,
    initial_row: Option<usize>,
) -> Result<(Array2<f64>, Vec<CovariateSegment>), EventHistoryError> {
    let n_cov = base.ncols();
    let base_rows = base.nrows();
    let mut table = Array2::<f64>::zeros((base_rows + future.len(), n_cov));
    table.slice_mut(ndarray::s![..base_rows, ..]).assign(&base);
    let mut segments: Vec<CovariateSegment> = Vec::new();
    if future.first().is_none_or(|s| s.start > window_start) {
        let row = initial_row.ok_or_else(|| EventHistoryError::InvalidInput {
            reason: format!(
                "the covariate path must cover the window from its start {window_start}: the first future segment starts at {:?}",
                future.first().map(|s| s.start)
            ),
        })?;
        segments.push(CovariateSegment {
            start: window_start,
            row,
        });
    }
    for (i, segment) in future.iter().enumerate() {
        for (j, value) in segment.covariates.iter().enumerate() {
            table[[base_rows + i, j]] = *value;
        }
        let start = segment.start.max(window_start);
        // Two segments at or before the start collapse onto the later one.
        if let Some(last) = segments.last_mut()
            && last.start == start
        {
            last.row = base_rows + i;
        } else {
            segments.push(CovariateSegment {
                start,
                row: base_rows + i,
            });
        }
    }
    Ok((table, segments))
}

/// Forecast one training subject beyond its observed exit.
pub fn forecast(
    fit: &EventHistoryFit,
    cohort: &EventHistoryCohort,
    request: &ForecastRequest<'_>,
) -> Result<Forecast, EventHistoryError> {
    forecast_on_table(
        fit,
        cohort,
        cohort.covariates.view(),
        request.history,
        request.stratum,
        request.horizons,
        request.future,
    )
}

/// Forecast any history beyond its exit: a subject that was not in the
/// training cohort, or a training subject's [`SubjectHistory::prefix`] at an
/// assessment time. The history's covariate segments index the request's
/// own covariate rows, which must be laid out in the cohort's columns and
/// level codes. The training subjects' rows are never consulted, so a
/// serving artifact needs the fit, not the cohort's histories.
pub fn forecast_history(
    fit: &EventHistoryFit,
    cohort: &EventHistoryCohort,
    request: &HistoryForecastRequest<'_>,
) -> Result<Forecast, EventHistoryError> {
    if request.covariates.ncols() != cohort.covariates.ncols() {
        return Err(EventHistoryError::InvalidInput {
            reason: format!(
                "the history's covariate rows have {} columns, the fit's cohort {}",
                request.covariates.ncols(),
                cohort.covariates.ncols()
            ),
        });
    }
    forecast_on_table(
        fit,
        cohort,
        request.covariates,
        request.history,
        request.stratum,
        request.horizons,
        request.future,
    )
}

fn forecast_on_table(
    fit: &EventHistoryFit,
    cohort: &EventHistoryCohort,
    table: ArrayView2<'_, f64>,
    history: &SubjectHistory,
    stratum: usize,
    horizons: &[f64],
    future: &[FutureSegment],
) -> Result<Forecast, EventHistoryError> {
    let marks = fit.marks();
    let kinds = &cohort.mark_kinds;
    if kinds.len() != marks || fit.mark_kinds != *kinds {
        return Err(EventHistoryError::InvalidInput {
            reason: "the forecast cohort's mark kinds differ from the fit's".to_string(),
        });
    }
    fit.risk_set_normaliser_at(stratum, history.entry)?;
    validate_horizons(horizons, history.exit)?;
    fit.risk_set_normaliser_at(stratum, horizons[horizons.len() - 1])?;
    let last_horizon = horizons[horizons.len() - 1];
    validate_future(cohort, future, last_horizon)?;
    // A subject whose follow-up ended with a terminal event has no future:
    // its survival is zero and nothing more can happen.
    if history.terminal_event(kinds).is_some() {
        return Ok(Forecast {
            horizons: horizons.to_vec(),
            survival: vec![0.0; horizons.len()],
            expected_counts: Array2::zeros((horizons.len(), marks)),
            survival_error: vec![0.0; horizons.len()],
            expected_count_errors: Array2::zeros((horizons.len(), marks)),
        });
    }
    let opening = Opening {
        table,
        history: history.clone(),
        compensated: vec![true; marks],
    };
    let (table, segments) = future_table(
        table,
        future,
        history.exit,
        Some(history.covariate_row_at(history.exit, false)),
    )?;
    let at_risk: Vec<bool> = (0..marks)
        .map(|d| match kinds[d] {
            MarkKind::Recurrent | MarkKind::Terminal => true,
            MarkKind::Once => !history.events.iter().any(|e| e.mark == d),
        })
        .collect();
    run_window(Window {
        fit,
        cohort,
        stratum,
        opening: Some(opening),
        start: history.exit,
        horizons,
        segments,
        table,
        at_risk,
        label: format!("{}::forecast", history.id),
    })
}

/// Forecast a subject with no observed history from covariate values alone.
pub fn population_forecast(
    fit: &EventHistoryFit,
    cohort: &EventHistoryCohort,
    request: &PopulationForecastRequest<'_>,
) -> Result<Forecast, EventHistoryError> {
    let marks = fit.marks();
    if cohort.mark_kinds.len() != marks || fit.mark_kinds != cohort.mark_kinds {
        return Err(EventHistoryError::InvalidInput {
            reason: "the forecast cohort's mark kinds differ from the fit's".to_string(),
        });
    }
    if !request.start.is_finite() {
        return Err(EventHistoryError::InvalidInput {
            reason: "the window's start must be finite".to_string(),
        });
    }
    fit.risk_set_normaliser_at(request.stratum, request.start)?;
    validate_horizons(request.horizons, request.start)?;
    fit.risk_set_normaliser_at(request.stratum, request.horizons[request.horizons.len() - 1])?;
    let last_horizon = request.horizons[request.horizons.len() - 1];
    validate_future(cohort, request.future, last_horizon)?;
    let (table, segments) = future_table(
        cohort.covariates.view(),
        request.future,
        request.start,
        None,
    )?;
    let at_risk = vec![true; marks];
    // Population forecasts describe people alive and free of once-only
    // diagnoses at start. Under reference centring they enter from that
    // selected reference law, not a fresh stationary draw at the late time.
    // Recurrent events are unobserved here and are integrated out.
    let opening = if let Some(snapshot) = fit.centring.as_ref() {
        let origin = snapshot.grid.times[0];
        if request.start > origin {
            let history = SubjectHistory {
                id: "reference::entry".to_string(), entry: origin, exit: request.start,
                events: Vec::new(),
                segments: vec![CovariateSegment { start: origin, row: request.stratum }],
            };
            let compensated: Vec<bool> = fit.mark_kinds.iter().map(|kind| *kind != MarkKind::Recurrent).collect();
            Some(Opening { table: snapshot.profiles.view(), history, compensated })
        } else { None }
    } else { None };
    run_window(Window {
        fit,
        cohort,
        stratum: request.stratum,
        opening,
        start: request.start,
        horizons: request.horizons,
        segments,
        table,
        at_risk,
        label: "population::forecast".to_string(),
    })
}

/// Predictive PIT of every spell of a subject's follow-up, in time order:
/// one per event, and one for the censored tail when the follow-up did not
/// end with an event. The event spells carry the predictive mark
/// probabilities at the event.
pub fn predictive_pit(
    fit: &EventHistoryFit,
    cohort: &EventHistoryCohort,
    history: &SubjectHistory,
    stratum: usize,
) -> Result<Vec<SpellPit>, EventHistoryError> {
    let marks = fit.marks();
    let kinds = &cohort.mark_kinds;
    let (loadings, rates) = latent_parameters(fit);
    let nodes = single_subject_nodes(fit, cohort, cohort.covariates.view(), history)?;
    let eta0 = node_eta0(fit, nodes.node_data.view())?;
    let subject = &nodes.subjects[0];
    let normaliser = forecast_normaliser(fit, stratum, &subject.times)?;
    let chronology = spells(
        &SubjectInputs {
            nodes: subject,
            eta0: &eta0,
            loadings: &loadings,
            rates: &rates,
            time_scale: fit.time_scale,
            gh: fit.family.gauss_hermite(),
            continuation_gap: 0.0,
            designs: None,
            log_normaliser: normaliser.as_deref(),
        },
        &vec![true; marks],
    )?;
    let spell_pit = |log_survival: f64, t: f64| -> Result<f64, EventHistoryError> {
        let pit = -log_survival.exp_m1();
        let slack = 64.0 * f64::EPSILON;
        if !pit.is_finite() || pit < -slack || pit > 1.0 + slack {
            return Err(EventHistoryError::NumericalFailure {
                reason: format!(
                    "subject {:?}: predictive survival to {t} is {}, outside [0, 1]",
                    history.id,
                    log_survival.exp()
                ),
            });
        }
        Ok(pit.clamp(0.0, 1.0))
    };
    let mut pits = Vec::with_capacity(chronology.len());
    for spell in chronology {
        // The tail: exposure after the last event (or the whole follow-up of a
        // subject without events) that ended at the exit without an event. Its
        // PIT is a censored draw — the uniform the model assigns to the spell
        // exceeds this value — and it is what makes the distance below a
        // statement about the model rather than about the censoring.
        let Some(intensities) = spell.intensities else {
            pits.push(SpellPit {
                time: history.exit,
                observed: false,
                pit: spell_pit(spell.log_survival, history.exit)?,
                marks: Vec::new(),
                mark_probabilities: vec![0.0; marks],
            });
            continue;
        };
        let n = spell.node;
        let t = subject.times[n];
        let pit = spell_pit(spell.log_survival, t)?;
        let at_risk: Vec<f64> = (0..marks)
            .map(|d| {
                if history.at_risk(d, t, kinds) {
                    intensities[d]
                } else {
                    0.0
                }
            })
            .collect();
        let total: f64 = at_risk.iter().sum();
        let mark_probabilities: Vec<f64> = if total > 0.0 {
            at_risk.iter().map(|v| v / total).collect()
        } else {
            vec![0.0; marks]
        };
        let mut fired = Vec::new();
        for d in 0..marks {
            let copies = subject.counts[[n, d]].round() as usize;
            fired.extend(std::iter::repeat_n(d, copies));
        }
        pits.push(SpellPit {
            time: t,
            observed: true,
            pit,
            marks: fired,
            mark_probabilities,
        });
    }
    Ok(pits)
}

/// Distance of the predictive PIT distribution from the uniform law,
/// estimated over event and censored spells alike: the largest gap between
/// the Kaplan–Meier estimate of the PIT distribution — an event spell is an
/// observation of its uniform, a censored spell is that uniform observed to
/// exceed the value emitted — and the uniform law, over the range the spells
/// cover. Under the model a spell's uniform exceeds its censored value
/// independently of the uniform itself given the history (the censoring on
/// the PIT scale is a function of the history and the exit alone), which is
/// what makes the estimate consistent. Without censoring it is the ordinary
/// Kolmogorov–Smirnov distance. `None` for
/// no spells. With parameters estimated from the same data it is a summary,
/// not a calibrated test.
pub fn pit_uniform_distance(pits: &[SpellPit]) -> Option<f64> {
    if pits.is_empty() {
        return None;
    }
    // Ties: an event at a value is counted before a censoring at the same
    // value, which leaves the censored spell in the risk set of the event.
    let mut spells: Vec<(f64, bool)> = pits.iter().map(|p| (p.pit, p.observed)).collect();
    spells.sort_by(|a, b| a.0.total_cmp(&b.0).then(b.1.cmp(&a.1)));
    let n = spells.len();
    let last_value = spells[n - 1].0;
    let mut distance = 0.0_f64;
    let mut survival = 1.0_f64;
    let mut i = 0usize;
    while i < n {
        let value = spells[i].0;
        let at_risk = (n - i) as f64;
        let mut events = 0usize;
        let mut j = i;
        while j < n && spells[j].0 == value {
            events += usize::from(spells[j].1);
            j += 1;
        }
        if events > 0 {
            let before = 1.0 - survival;
            survival *= 1.0 - events as f64 / at_risk;
            let after = 1.0 - survival;
            distance = distance
                .max((before - value).abs())
                .max((after - value).abs());
        }
        i = j;
    }
    // Between the last jump and the largest value the estimate is flat while
    // the uniform keeps rising: the gap at the end of the covered range.
    distance = distance.max((1.0 - survival - last_value).abs());
    Some(distance)
}
