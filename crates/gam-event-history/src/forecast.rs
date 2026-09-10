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
//! integral, and the identity `Σ_{d terminal} F_d(h) = 1 − S(h)` holds to
//! quadrature accuracy.
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
//! in the spell | history)`, which the filter yields as the product of the
//! normalisers of the zero-count nodes in the spell. Under the model, the
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

use super::chain::Grid;
use super::cohort::{
    CohortNodes, CovariateSegment, EventHistoryCohort, EventHistoryError, MarkKind, SubjectHistory,
    SubjectNodes, cell_rule, expand_nodes, mesh_cells,
};
use super::family::EventHistoryFit;
use super::marginal::{
    ForwardPass, SubjectInputs, expected_intensities, forward_filter, latent_state_moments,
};
use gam_terms::smooth::build_term_collection_design;
use ndarray::{Array1, Array2, ArrayView2};

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
/// when terminal, its first-occurrence probability when once-only).
#[derive(Clone, Debug)]
pub struct Forecast {
    pub horizons: Vec<f64>,
    pub survival: Vec<f64>,
    pub expected_counts: Array2<f64>,
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

/// The filtered latent state after a subject's observed history, whose
/// covariate segments index `table`.
fn observed_state(
    fit: &EventHistoryFit,
    cohort: &EventHistoryCohort,
    table: ArrayView2<'_, f64>,
    history: &SubjectHistory,
    stratum: usize,
    loadings: &[f64],
    rates: &[f64],
    compensated: &[bool],
) -> Result<LatentState, EventHistoryError> {
    let observed = single_subject_nodes(fit, cohort, table, history)?;
    let eta0 = node_eta0(fit, observed.node_data.view())?;
    let normaliser = forecast_normaliser(fit, stratum, &observed.subjects[0].times)?;
    let mut pass = forward_filter(
        &SubjectInputs {
            nodes: &observed.subjects[0],
            eta0: &eta0,
            loadings,
            rates,
            time_scale: fit.time_scale,
            gh: fit.family.gauss_hermite(),
            continuation_gap: 0.0,
            designs: None,
            log_normaliser: normaliser.as_deref(),
        },
        None,
        compensated,
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
/// latent state it continues from (`None` for the stationary prior), and
/// which marks the subject is still at risk for.
struct Window<'a> {
    fit: &'a EventHistoryFit,
    cohort: &'a EventHistoryCohort,
    stratum: usize,
    initial: Option<&'a LatentState>,
    start: f64,
    horizons: &'a [f64],
    segments: Vec<CovariateSegment>,
    /// The cohort's covariate table extended by the window's own rows.
    table: Array2<f64>,
    at_risk: Vec<bool>,
    label: String,
}

/// The killed-process integration of one forecast window.
fn run_window(window: Window<'_>) -> Result<Forecast, EventHistoryError> {
    let Window {
        fit,
        cohort,
        stratum,
        initial,
        start,
        horizons,
        segments,
        table,
        at_risk,
        label,
    } = window;
    let at_risk = at_risk.as_slice();
    let marks = fit.marks();
    let atoms = fit.rank();
    let kinds = &cohort.mark_kinds;
    let n_cov = cohort.covariates.ncols();
    let (loadings, rates) = latent_parameters(fit);
    let last_horizon = horizons[horizons.len() - 1];
    // Every horizon is a mesh breakpoint, so cell ends land on horizons.
    let mut segments = segments;
    for &h in &horizons[..horizons.len() - 1] {
        if !segments.iter().any(|s| s.start == h) {
            let row = segments
                .iter()
                .rev()
                .find(|s| s.start <= h)
                .map(|s| s.row)
                .expect("a segment starts at the window's start");
            segments.push(CovariateSegment { start: h, row });
        }
    }
    segments.sort_by(|a, b| a.start.total_cmp(&b.start));
    let pseudo = SubjectHistory {
        id: label,
        entry: start,
        exit: last_horizon,
        events: Vec::new(),
        segments,
    };
    let (gl_nodes, gl_weights) = gam_math::special::gauss_legendre(fit.quadrature_order);
    let cells = mesh_cells(&pseudo, false, fit.mesh_refinement);
    let q = fit.quadrature_order;
    // Evaluation points: the outer nodes of every cell, then the inner nodes
    // of every outer node's own chronological rule from the cell's start.
    let mut point_times: Vec<f64> = Vec::new();
    let mut point_rows: Vec<usize> = Vec::new();
    let mut outer: Vec<Vec<(f64, f64, usize)>> = Vec::with_capacity(cells.len());
    let mut inner: Vec<Vec<Vec<(f64, f64, usize)>>> = Vec::with_capacity(cells.len());
    let mut push_point = |t: f64| -> usize {
        point_times.push(t);
        point_rows.push(pseudo.covariate_row_at(t, false));
        point_times.len() - 1
    };
    for &(left, right) in &cells {
        let outer_rule: Vec<(f64, f64)> = cell_rule(left, right, &gl_nodes, &gl_weights).collect();
        let outer_nodes: Vec<(f64, f64, usize)> = outer_rule
            .iter()
            .map(|&(t, w)| (t, w, push_point(t)))
            .collect();
        let mut inner_cell = Vec::with_capacity(q);
        for &(t_j, _, _) in &outer_nodes {
            let rule: Vec<(f64, f64)> = cell_rule(left, t_j, &gl_nodes, &gl_weights).collect();
            let chain: Vec<(f64, f64, usize)> =
                rule.iter().map(|&(s, v)| (s, v, push_point(s))).collect();
            inner_cell.push(chain);
        }
        outer.push(outer_nodes);
        inner.push(inner_cell);
    }
    let mut rows = Array2::<f64>::zeros((point_times.len(), n_cov + 1));
    for (i, (&t, &row)) in point_times.iter().zip(point_rows.iter()).enumerate() {
        for j in 0..n_cov {
            rows[[i, j]] = table[[row, j]];
        }
        rows[[i, n_cov]] = t;
    }
    let eta0 = node_eta0(fit, rows.view())?;
    let eta_at = |point: usize| -> &[f64] { &eta0[point * marks..(point + 1) * marks] };
    let gh = fit.family.gauss_hermite();

    // One killed run under a killing set: the survival at the end of every
    // cell and the sub-density `m_d(t) = S(t) E[λ_d(t)]` at every outer node.
    let killed_run = |killing: &[bool]| -> Result<(Vec<f64>, Vec<Vec<f64>>), EventHistoryError> {
        let exposed: Vec<bool> = (0..marks).map(|d| killing[d] && at_risk[d]).collect();
        let mut state: Option<LatentState> = initial.cloned();
        let mut log_s = 0.0_f64;
        let mut cell_log_s = Vec::with_capacity(cells.len());
        let mut sub_density: Vec<Vec<f64>> = Vec::with_capacity(cells.len() * q);
        let filter = |state: &Option<LatentState>,
                      times: &[f64],
                      weights: &[f64],
                      eta: &[f64]|
         -> Result<ForwardPass<f64>, EventHistoryError> {
            let nodes = future_chain(times, weights, &exposed, marks);
            let normaliser = forecast_normaliser(fit, stratum, times)?;
            forward_filter(
                &SubjectInputs {
                    nodes: &nodes,
                    eta0: eta,
                    loadings: &loadings,
                    rates: &rates,
                    time_scale: fit.time_scale,
                    gh,
                    continuation_gap: state.as_ref().map_or(0.0, |s| times[0] - s.time),
                    designs: None,
                    log_normaliser: normaliser.as_deref(),
                },
                state.as_ref().map(|s| (&s.grid, s.alpha.as_slice())),
                &exposed,
            )
        };
        for (c, outer_nodes) in outer.iter().enumerate() {
            for (j, &(t_j, _, point_j)) in outer_nodes.iter().enumerate() {
                let chain = &inner[c][j];
                let mut times: Vec<f64> = chain.iter().map(|&(s, _, _)| s).collect();
                let mut weights: Vec<f64> = chain.iter().map(|&(_, v, _)| v).collect();
                times.push(t_j);
                weights.push(0.0);
                let mut chain_eta = Vec::with_capacity(times.len() * marks);
                for &(_, _, point) in chain {
                    chain_eta.extend_from_slice(eta_at(point));
                }
                chain_eta.extend_from_slice(eta_at(point_j));
                let pass = filter(&state, &times, &weights, &chain_eta)?;
                let log_s_j: f64 = log_s + pass.log_normalisers.iter().sum::<f64>();
                let last = times.len() - 1;
                let at_j = forecast_normaliser(fit, stratum, &[t_j])?;
                let intensities = expected_intensities(
                    &pass.grids[last],
                    &pass.predicted[last],
                    eta_at(point_j),
                    &loadings,
                    at_j.as_deref(),
                    marks,
                    atoms,
                );
                sub_density.push(
                    (0..marks)
                        .map(|d| {
                            if at_risk[d] {
                                log_s_j.exp() * intensities[d]
                            } else {
                                0.0
                            }
                        })
                        .collect(),
                );
            }
            // Advance the state across the cell along its own rule.
            let times: Vec<f64> = outer_nodes.iter().map(|&(t, _, _)| t).collect();
            let weights: Vec<f64> = outer_nodes.iter().map(|&(_, w, _)| w).collect();
            let mut chain_eta = Vec::with_capacity(times.len() * marks);
            for &(_, _, point) in outer_nodes {
                chain_eta.extend_from_slice(eta_at(point));
            }
            let mut pass = filter(&state, &times, &weights, &chain_eta)?;
            log_s += pass.log_normalisers.iter().sum::<f64>();
            state = Some(LatentState {
                grid: pass.grids.pop().expect("cell has nodes"),
                alpha: pass.alpha.pop().expect("cell has nodes"),
                time: times[times.len() - 1],
            });
            cell_log_s.push(log_s);
        }
        Ok((cell_log_s, sub_density))
    };

    let terminal: Vec<bool> = kinds.iter().map(|k| *k == MarkKind::Terminal).collect();
    let (cell_log_s, base_density) = killed_run(&terminal)?;
    // Once-only marks are killed by their own hazard as well.
    let mut once_density: Vec<Option<Vec<Vec<f64>>>> = vec![None; marks];
    for d in 0..marks {
        if kinds[d] == MarkKind::Once && at_risk[d] {
            let mut killing = terminal.clone();
            killing[d] = true;
            once_density[d] = Some(killed_run(&killing)?.1);
        }
    }
    let n_h = horizons.len();
    let mut survival = vec![0.0; n_h];
    let mut expected = Array2::<f64>::zeros((n_h, marks));
    let mut counts = vec![0.0; marks];
    let mut horizon = 0usize;
    let mut point = 0usize;
    for (c, &(_, right)) in cells.iter().enumerate() {
        for &(_, w_j, _) in outer[c].iter() {
            for d in 0..marks {
                let density = match &once_density[d] {
                    Some(run) => run[point][d],
                    None => base_density[point][d],
                };
                counts[d] += w_j * density;
            }
            point += 1;
        }
        if horizon < n_h && right == horizons[horizon] {
            survival[horizon] = cell_log_s[c].exp();
            for d in 0..marks {
                expected[[horizon, d]] = counts[d];
            }
            horizon += 1;
        }
    }
    if horizon != n_h {
        return Err(EventHistoryError::NumericalFailure {
            reason: format!("forecast mesh reached {horizon} of {n_h} horizons"),
        });
    }
    Ok(Forecast {
        horizons: horizons.to_vec(),
        survival,
        expected_counts: expected,
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
        });
    }
    let (loadings, rates) = latent_parameters(fit);
    let state = observed_state(fit, cohort, table, history, stratum, &loadings, &rates, &vec![true; marks])?;
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
        initial: Some(&state),
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
    let initial = if let Some(snapshot) = fit.centring.as_ref() {
        let origin = snapshot.grid.times[0];
        if request.start > origin {
            let history = SubjectHistory {
                id: "reference::entry".to_string(), entry: origin, exit: request.start,
                events: Vec::new(),
                segments: vec![CovariateSegment { start: origin, row: request.stratum }],
            };
            let (loadings, rates) = latent_parameters(fit);
            let compensated: Vec<bool> = fit.mark_kinds.iter().map(|kind| *kind != MarkKind::Recurrent).collect();
            Some(observed_state(fit, cohort, snapshot.profiles.view(), &history,
                request.stratum, &loadings, &rates, &compensated)?)
        } else { None }
    } else { None };
    run_window(Window {
        fit,
        cohort,
        stratum: request.stratum,
        initial: initial.as_ref(),
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
    let atoms = fit.rank();
    let kinds = &cohort.mark_kinds;
    let (loadings, rates) = latent_parameters(fit);
    let nodes = single_subject_nodes(fit, cohort, cohort.covariates.view(), history)?;
    let eta0 = node_eta0(fit, nodes.node_data.view())?;
    let subject = &nodes.subjects[0];
    let normaliser = forecast_normaliser(fit, stratum, &subject.times)?;
    let pass = forward_filter(
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
        None,
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
    let mut pits = Vec::new();
    let mut log_survival: f64 = 0.0;
    let mut open = false;
    for n in 0..subject.len() {
        if !subject.is_event(n) {
            log_survival += pass.log_normalisers[n];
            open = true;
            continue;
        }
        let t = subject.times[n];
        let pit = spell_pit(log_survival, t)?;
        let intensities = expected_intensities(
            &pass.grids[n],
            &pass.predicted[n],
            &eta0[n * marks..(n + 1) * marks],
            &loadings,
            normaliser
                .as_deref()
                .map(|m| &m[n * marks..(n + 1) * marks]),
            marks,
            atoms,
        );
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
        log_survival = 0.0;
        open = false;
    }
    // The tail: exposure after the last event (or the whole follow-up of a
    // subject without events) that ended at the exit without an event. Its
    // PIT is a censored draw — the uniform the model assigns to the spell
    // exceeds this value — and it is what makes the distance below a
    // statement about the model rather than about the censoring.
    if open {
        pits.push(SpellPit {
            time: history.exit,
            observed: false,
            pit: spell_pit(log_survival, history.exit)?,
            marks: Vec::new(),
            mark_probabilities: vec![0.0; marks],
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
/// Kolmogorov–Smirnov distance ([`kolmogorov_smirnov_uniform`]). `None` for
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

/// Kolmogorov–Smirnov distance of an uncensored PIT sample from the uniform
/// law, or `None` for an empty sample. This is the right summary only when
/// every spell ended with an event; see [`pit_uniform_distance`] for the
/// general case, which this equals when nothing is censored.
pub fn kolmogorov_smirnov_uniform(pits: &[f64]) -> Option<f64> {
    if pits.is_empty() {
        return None;
    }
    let mut sorted = pits.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let n = sorted.len() as f64;
    let mut distance = 0.0_f64;
    for (i, &u) in sorted.iter().enumerate() {
        let lower = i as f64 / n;
        let upper = (i + 1) as f64 / n;
        distance = distance.max((u - lower).abs()).max((upper - u).abs());
    }
    Some(distance)
}

/// Convenience: the fitted per-mark linear predictor on the training nodes.
pub fn training_eta(fit: &EventHistoryFit) -> Array2<f64> {
    let marks = fit.marks();
    let total = fit.nodes.total_nodes;
    let mut out = Array2::<f64>::zeros((total, marks));
    for d in 0..marks {
        let eta: &Array1<f64> = fit.mark_eta(d);
        for row in 0..total {
            out[[row, d]] = eta[row];
        }
    }
    out
}
