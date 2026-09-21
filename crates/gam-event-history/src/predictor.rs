//! The predictor a fit publishes, and the posterior its forecasts average
//! over.
//!
//! A forecast is a probability under the model, and the model's global
//! parameters are uncertain. The survival to a horizon is `E_θ[S_θ(h) | data]`:
//! under every parameter state `θ` (baselines, loadings, rates) the killed
//! latent process is filtered and integrated, and only the final probabilities
//! are averaged. Evaluating at the one stored mode gives a different number
//! whenever the forecast is nonlinear in `θ`: under an exponential posterior of
//! mean one for a constant rate, the one-unit event probability is ½, while the
//! rate at its mean gives `1 − e⁻¹`.
//!
//! The global posterior is the fit's Laplace approximation `θ ~ N(θ̂, V)` over
//! every block: the mark coefficients, the loadings and the free rate
//! coordinates. This is a Laplace approximation of the posterior. `V` is the
//! smoothing-corrected covariance when the fit estimated smoothing strengths.
//! When there were none to estimate, the conditional covariance is the
//! posterior. A fit that selected strengths but records why it minted no
//! correction is refused, because its conditional covariance would understate
//! the spread. Every
//! node of the integration rule is a complete parameter state that carries its
//! own reference law, evaluated at its own coefficients, so a baseline is never
//! divided by another state's normaliser.
//!
//! The rule is a dimension-adaptive sparse grid (Gerstner and Griebel, 2003)
//! over the eigen-directions of `V`. It is built from odd-order Gauss-Hermite
//! rules whose middle node is the mode. Each index contributes a difference of
//! successive one-dimensional rules, so the contributions of the indices on the
//! frontier are what one more refinement could still move. Every state's
//! forecast carries the integration error its own window measured, and the
//! rule carries those errors through its weights in absolute value. The rule
//! stops when, for every returned entry, the summed magnitude of the frontier
//! contributions, its estimate of what one more refinement would still move, no
//! longer exceeds that integration error, or the combination's rounding band
//! where that is larger, and it reports the total of the three.
//! No tolerance is chosen: the posterior integral is resolved to the accuracy
//! of the probabilities it averages, and no further. A parameter state the
//! model cannot evaluate is an error, never a skipped node.

use super::chain::GaussHermite;
use super::cohort::{EventHistoryCohort, EventHistoryError, MarkKind};
use super::family::{EventHistoryFit, fit_event_history_formulas, rate_from_chart};
use super::preserve::{ReferenceGrid, ReferenceStrata, stratum_normalisers};
use gam_linalg::roundoff::{accumulation_band, symmetric_spectrum_rounding_band};
use gam_model_api::families::custom_family::BlockwiseFitOptions;
use gam_terms::smooth::{TermCollectionSpec, build_term_collection_design};
use ndarray::{Array1, Array2, ArrayView2};
use rayon::prelude::*;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex};

/// A node of the sparse rule, as its offsets from the mode: per direction, the
/// one-dimensional level and node index. The mode is the empty key.
type PointKey = Vec<(usize, usize, usize)>;

/// A multi-index of the sparse rule: the directions refined past the mode, in
/// direction order, each with its level (at least two).
type Index = Vec<(usize, usize)>;

/// A fitted event-history model as prediction reads it. It holds the frozen
/// schema and bases, the numerical settings the fit was certified at, the
/// reference law's grid and profiles, and the posterior of every coefficient.
/// It holds no training histories.
pub struct EventHistoryPredictor {
    pub mark_names: Vec<String>,
    pub mark_kinds: Vec<MarkKind>,
    pub covariate_names: Vec<String>,
    pub covariate_levels: Vec<Vec<String>>,
    frozen_specs: Vec<TermCollectionSpec>,
    /// Gauss-Legendre order per mesh cell.
    pub(crate) quadrature_order: usize,
    pub(crate) mesh_refinement: usize,
    pub(crate) time_scale: f64,
    pub(crate) gh: Arc<GaussHermite>,
    posterior: LaplacePosterior,
    reference: Option<ReferenceLaw>,
    /// Parameter states already evaluated, by node. A state is a function of
    /// its node alone, so every forecast reads the same one.
    states: Mutex<BTreeMap<PointKey, Arc<ParameterState>>>,
}

/// The Laplace posterior of every coefficient, in the fit's block order.
struct LaplacePosterior {
    /// `θ̂`: every mark block, then the latent block.
    mode: Vec<f64>,
    /// The width of every mark block.
    widths: Vec<usize>,
    atoms: usize,
    rate_band: (f64, f64),
    /// Per atom, its held rate, or `None` when the rate is a coefficient.
    held_rates: Vec<Option<f64>>,
    /// The resolved eigen-directions of `V`: standard deviation and unit vector.
    directions: Vec<(f64, Vec<f64>)>,
}

/// The reference population's grid and covariate profiles, with every
/// stratum's design rows at every grid time, per mark.
struct ReferenceLaw {
    grid: ReferenceGrid,
    profiles: Array2<f64>,
    designs: Vec<Arc<Array2<f64>>>,
    offsets: Vec<Array1<f64>>,
}

impl ReferenceLaw {
    /// The reference law on `grid`: every stratum's profile at every grid time
    /// (covariate columns then time), through each mark's frozen basis.
    fn on(
        grid: ReferenceGrid,
        profiles: Array2<f64>,
        frozen_specs: &[TermCollectionSpec],
    ) -> Result<Self, EventHistoryError> {
        let columns = profiles.ncols();
        let mut rows = Array2::<f64>::zeros((profiles.nrows() * grid.len(), columns + 1));
        for s in 0..profiles.nrows() {
            for (n, &time) in grid.times.iter().enumerate() {
                let row = s * grid.len() + n;
                for c in 0..columns {
                    rows[[row, c]] = profiles[[s, c]];
                }
                rows[[row, columns]] = time;
            }
        }
        let mut designs = Vec::with_capacity(frozen_specs.len());
        let mut offsets = Vec::with_capacity(frozen_specs.len());
        for (d, spec) in frozen_specs.iter().enumerate() {
            let design = build_term_collection_design(rows.view(), spec).map_err(|error| {
                EventHistoryError::Fit {
                    reason: format!("reference design for mark {d}: {error}"),
                }
            })?;
            designs.push(
                design
                    .design
                    .try_to_dense_arc("event-history reference design")
                    .map_err(|error| EventHistoryError::Fit {
                        reason: error.to_string(),
                    })?,
            );
            offsets.push(design.affine_offset.clone());
        }
        Ok(Self {
            grid,
            profiles,
            designs,
            offsets,
        })
    }
}

/// One global parameter state: the coefficients of every mark, the loadings
/// (index `d * atoms + k`), the dimensionless rates, and the reference law's
/// log normaliser at those coefficients.
pub(crate) struct ParameterState {
    betas: Vec<Array1<f64>>,
    pub(crate) loadings: Vec<f64>,
    pub(crate) rates: Vec<f64>,
    /// `log M_d` on the reference grid, `strata · grid · marks`; `None` under
    /// the stationary prior's centring.
    log_normaliser: Option<Vec<f64>>,
}

/// What one parameter state's forecast returns to the posterior rule: its
/// values and the integration error measured on each.
pub(crate) struct Evaluated {
    pub(crate) values: Vec<f64>,
    pub(crate) errors: Vec<f64>,
}

/// A posterior-predictive average of the vector a forecast returns.
pub(crate) struct Averaged {
    pub(crate) values: Vec<f64>,
    /// The checked error of every value: the frontier contributions one more
    /// refinement could still move, the integration error the states measured
    /// carried through the rule, and the rounding band of the combination.
    pub(crate) errors: Vec<f64>,
    /// Parameter states the rule evaluated.
    pub(crate) evaluations: usize,
}

/// A fitted event-history model with the predictor its forecasts read, built
/// only when a forecast asks for it. A fit whose posterior cannot be averaged
/// over, such as one that selected smoothing strengths but minted no smoothing
/// correction, still fits and reports. Only its forecasts refuse, with the
/// predictor's typed reason.
pub struct FittedEventHistory {
    pub fit: EventHistoryFit,
    pub cohort: EventHistoryCohort,
    predictor: Mutex<Option<Arc<EventHistoryPredictor>>>,
}

impl FittedEventHistory {
    /// Fit the cohort. No part of forecasting runs here.
    pub fn fit<F: AsRef<str>>(
        mut cohort: EventHistoryCohort,
        formulas: &[F],
        options: BlockwiseFitOptions,
        reference: Option<ReferenceStrata>,
    ) -> Result<Self, EventHistoryError> {
        let fit = fit_event_history_formulas(&mut cohort, formulas, options, reference)?;
        Ok(Self {
            fit,
            cohort,
            predictor: Mutex::new(None),
        })
    }

    /// The predictor every forecast of this fit reads, built on the first
    /// request. A refusal is returned to that request and to every later one.
    pub fn predictor(&self) -> Result<Arc<EventHistoryPredictor>, EventHistoryError> {
        if let Some(predictor) = self.predictor.lock().ok().and_then(|held| held.clone()) {
            return Ok(predictor);
        }
        let predictor = Arc::new(EventHistoryPredictor::from_fit(&self.fit, &self.cohort)?);
        if let Ok(mut held) = self.predictor.lock() {
            held.get_or_insert_with(|| Arc::clone(&predictor));
        }
        Ok(predictor)
    }
}

/// Everything a predictor is assembled from.
struct PredictorParts {
    mark_names: Vec<String>,
    mark_kinds: Vec<MarkKind>,
    covariate_names: Vec<String>,
    covariate_levels: Vec<Vec<String>>,
    frozen_specs: Vec<TermCollectionSpec>,
    quadrature_order: usize,
    mesh_refinement: usize,
    time_scale: f64,
    gauss_hermite_order: usize,
    mode: Vec<f64>,
    widths: Vec<usize>,
    /// The resolved eigen-directions of the posterior covariance: standard
    /// deviation and unit vector.
    directions: Vec<(f64, Vec<f64>)>,
    atoms: usize,
    rate_band: (f64, f64),
    held_rates: Vec<Option<f64>>,
    reference: Option<(ReferenceGrid, Array2<f64>)>,
}

impl EventHistoryPredictor {
    /// The predictor of a fit: the schema of the cohort it was fitted on, its
    /// frozen bases and certified settings, its reference law, and the Laplace
    /// posterior of every coefficient.
    pub fn from_fit(
        fit: &EventHistoryFit,
        cohort: &EventHistoryCohort,
    ) -> Result<Self, EventHistoryError> {
        let marks = fit.marks();
        let atoms = fit.rank();
        let blocks = &fit.fit.block_states;
        if blocks.len() != marks + usize::from(atoms > 0) {
            return Err(EventHistoryError::InvalidInput {
                reason: format!(
                    "a fit of {marks} marks and {atoms} atoms carries {} coefficient blocks",
                    blocks.len()
                ),
            });
        }
        let rates = if atoms > 0 {
            fit.family.atom_rates(&blocks[marks].beta)
        } else {
            Vec::new()
        };
        let held_rates = fit
            .family
            .rate_held()
            .iter()
            .zip(rates)
            .map(|(&held, rate)| held.then_some(rate))
            .collect();
        // A fit that selected smoothing strengths but minted no correction says
        // why, and its conditional covariance would understate the spread the
        // forecasts average over, so it is refused rather than used. With no
        // strength to estimate there is nothing to correct, and the
        // conditional covariance is the posterior.
        let covariance = match (
            fit.fit.beta_covariance_corrected(),
            fit.fit.smoothing_correction_absence(),
        ) {
            (Some(corrected), _) => corrected,
            (None, Some(absence)) => {
                return Err(EventHistoryError::InvalidInput {
                    reason: format!(
                        "the fit selected smoothing strengths but retains no smoothing-uncertainty correction ({absence}), so its conditional covariance would understate the posterior that forecasts average over"
                    ),
                });
            }
            (None, None) => fit.fit.beta_covariance().ok_or_else(|| {
                EventHistoryError::InvalidInput {
                    reason: "the fit carries no posterior covariance to average forecasts over"
                        .to_string(),
                }
            })?,
        };
        Self::on_posterior(
            fit,
            cohort,
            blocks.iter().flat_map(|b| b.beta.iter().copied()).collect(),
            blocks[..marks].iter().map(|b| b.beta.len()).collect(),
            posterior_directions(covariance)?,
            held_rates,
        )
    }

    /// A fit's schema, frozen bases, settings and reference law, on a given
    /// posterior.
    fn on_posterior(
        fit: &EventHistoryFit,
        cohort: &EventHistoryCohort,
        mode: Vec<f64>,
        widths: Vec<usize>,
        directions: Vec<(f64, Vec<f64>)>,
        held_rates: Vec<Option<f64>>,
    ) -> Result<Self, EventHistoryError> {
        if cohort.mark_kinds != fit.mark_kinds || cohort.mark_names.len() != fit.marks() {
            return Err(EventHistoryError::InvalidInput {
                reason: "the cohort's marks differ from the fit's".to_string(),
            });
        }
        Self::assemble(PredictorParts {
            mark_names: cohort.mark_names.clone(),
            mark_kinds: fit.mark_kinds.clone(),
            covariate_names: cohort.covariate_names.clone(),
            covariate_levels: cohort.covariate_levels.clone(),
            frozen_specs: fit.frozen_specs.clone(),
            quadrature_order: fit.quadrature_order,
            mesh_refinement: fit.mesh_refinement,
            time_scale: fit.time_scale,
            gauss_hermite_order: fit.family.gauss_hermite_order(),
            mode,
            widths,
            directions,
            atoms: held_rates.len(),
            rate_band: fit.family.rate_band(),
            held_rates,
            reference: fit
                .centring
                .as_ref()
                .map(|snapshot| (snapshot.grid.clone(), snapshot.profiles.clone())),
        })
    }

    fn assemble(parts: PredictorParts) -> Result<Self, EventHistoryError> {
        let PredictorParts {
            mark_names,
            mark_kinds,
            covariate_names,
            covariate_levels,
            frozen_specs,
            quadrature_order,
            mesh_refinement,
            time_scale,
            gauss_hermite_order,
            mode,
            widths,
            directions,
            atoms,
            rate_band,
            held_rates,
            reference,
        } = parts;
        let marks = mark_kinds.len();
        let width = mode.len();
        let free_rates = held_rates.iter().filter(|held| held.is_none()).count();
        if mark_names.len() != marks
            || widths.len() != marks
            || frozen_specs.len() != marks
            || held_rates.len() != atoms
            || covariate_levels.len() != covariate_names.len()
            || widths.iter().sum::<usize>() + marks * atoms + free_rates != width
            || mode.iter().any(|x| !x.is_finite())
            || directions.iter().any(|(spread, vector)| {
                !(spread.is_finite() && *spread > 0.0)
                    || vector.len() != width
                    || vector.iter().any(|x| !x.is_finite())
            })
        {
            return Err(EventHistoryError::InvalidInput {
                reason: "a predictor needs one coefficient block and frozen basis per mark, one rate per atom, a finite mode, and finite posterior directions of positive spread across every coefficient".to_string(),
            });
        }
        let reference = reference
            .map(|(grid, profiles)| ReferenceLaw::on(grid, profiles, &frozen_specs))
            .transpose()?;
        Ok(Self {
            mark_names,
            mark_kinds,
            covariate_names,
            covariate_levels,
            frozen_specs,
            quadrature_order,
            mesh_refinement,
            time_scale,
            gh: Arc::new(GaussHermite::new(gauss_hermite_order)?),
            posterior: LaplacePosterior {
                mode,
                widths,
                atoms,
                rate_band,
                held_rates,
                directions,
            },
            reference,
            states: Mutex::new(BTreeMap::new()),
        })
    }

    pub fn marks(&self) -> usize {
        self.mark_kinds.len()
    }

    pub fn atoms(&self) -> usize {
        self.posterior.atoms
    }

    /// Refuse a stratum the model has no reference population for, or a time
    /// outside the reference law's interval.
    pub(crate) fn check_reference_time(
        &self,
        stratum: usize,
        t: f64,
    ) -> Result<(), EventHistoryError> {
        match &self.reference {
            None if stratum != 0 => Err(EventHistoryError::InvalidInput {
                reason: "a model without reference strata requires stratum zero".to_string(),
            }),
            None => Ok(()),
            Some(law) if stratum >= law.profiles.nrows() => Err(EventHistoryError::InvalidInput {
                reason: format!(
                    "reference stratum {stratum} is outside 0..{}",
                    law.profiles.nrows()
                ),
            }),
            Some(law) => {
                law.grid.locate(t)?;
                Ok(())
            }
        }
    }

    /// Where the reference population starts, with its strata's covariate
    /// profiles; `None` under the stationary prior's centring.
    pub(crate) fn reference_entry(&self) -> Option<(f64, ArrayView2<'_, f64>)> {
        self.reference
            .as_ref()
            .map(|law| (law.grid.times[0], law.profiles.view()))
    }

    /// Node log-intensities `η⁰` of every mark at a state, on a row matrix
    /// (covariate columns then time), index `row * marks + d`.
    pub(crate) fn eta0(
        &self,
        state: &ParameterState,
        rows: ArrayView2<'_, f64>,
    ) -> Result<Vec<f64>, EventHistoryError> {
        let betas: Vec<&Array1<f64>> = state.betas.iter().collect();
        design_eta0(&self.frozen_specs, &betas, rows)
    }

    /// The parameter state at a coefficient vector in the fit's block order,
    /// with its reference law evaluated at those coefficients.
    fn state_at(&self, theta: &[f64]) -> Result<ParameterState, EventHistoryError> {
        let marks = self.marks();
        let atoms = self.posterior.atoms;
        let mut offset = 0;
        let mut betas = Vec::with_capacity(marks);
        for &width in &self.posterior.widths {
            betas.push(Array1::from(theta[offset..offset + width].to_vec()));
            offset += width;
        }
        let loadings = theta[offset..offset + marks * atoms].to_vec();
        let mut slot = offset + marks * atoms;
        let mut rates = Vec::with_capacity(atoms);
        for held in &self.posterior.held_rates {
            match held {
                Some(rate) => rates.push(*rate),
                None => {
                    rates.push(rate_from_chart(self.posterior.rate_band, &theta[slot]));
                    slot += 1;
                }
            }
        }
        let log_normaliser = match &self.reference {
            None => None,
            Some(law) => {
                let nodes = law.grid.len();
                let strata = law.profiles.nrows();
                let mut out = Vec::with_capacity(strata * nodes * marks);
                for s in 0..strata {
                    let mut eta0 = Vec::with_capacity(nodes * marks);
                    for n in 0..nodes {
                        let row = s * nodes + n;
                        for (d, beta) in betas.iter().enumerate() {
                            let mut value = law.offsets[d][row];
                            for (j, x) in law.designs[d].row(row).iter().enumerate() {
                                value += x * beta[j];
                            }
                            eta0.push(value);
                        }
                    }
                    let normalisers = stratum_normalisers(
                        &law.grid,
                        &eta0,
                        &loadings,
                        &rates,
                        self.time_scale,
                        &self.gh,
                        &self.mark_kinds,
                        atoms,
                    )?;
                    out.extend(normalisers.log_normaliser);
                }
                Some(out)
            }
        };
        Ok(ParameterState {
            betas,
            loadings,
            rates,
            log_normaliser,
        })
    }

    /// The parameter state at one node of the rule, from the cache when an
    /// earlier forecast already evaluated it.
    fn state(
        &self,
        key: &PointKey,
        rules: &[NormalRule],
    ) -> Result<Arc<ParameterState>, EventHistoryError> {
        if let Some(state) = self
            .states
            .lock()
            .ok()
            .and_then(|cache| cache.get(key).cloned())
        {
            return Ok(state);
        }
        let mut theta = self.posterior.mode.clone();
        for &(direction, level, node) in key {
            let (spread, vector) = &self.posterior.directions[direction];
            let step = spread * rules[level - 1].nodes[node];
            for (coordinate, component) in theta.iter_mut().zip(vector) {
                *coordinate += step * component;
            }
        }
        let state = Arc::new(self.state_at(&theta)?);
        if let Ok(mut cache) = self.states.lock() {
            // The cache is a convenience across requests, never counted by
            // admission. It is emptied before it would hold more than the
            // materialisation budget.
            let held = (cache.len() + 1) * self.state_floats() * std::mem::size_of::<f64>();
            if held as f64 > Self::materialisation_budget() {
                cache.clear();
            }
            cache.entry(key.clone()).or_insert_with(|| Arc::clone(&state));
        }
        Ok(state)
    }

    /// Refuse a rule that would hold more parameter states and forecasts than
    /// this machine's materialisation budget allows. Every node of the request
    /// holds its parameter state and its values and errors. States cached by
    /// earlier requests are not counted, so whether a request is admitted never
    /// depends on what ran before it. The cache is emptied before it would pass
    /// the same budget, so a forecast holds at most twice the budget: the cache,
    /// plus this request's values and errors and the states it adds.
    fn check_budget(&self, nodes: usize, width: usize) -> Result<(), EventHistoryError> {
        let floats = nodes * (self.state_floats() + 2 * width);
        let bytes = floats as f64 * std::mem::size_of::<f64>() as f64;
        let budget = Self::materialisation_budget();
        if bytes > budget {
            return Err(EventHistoryError::NumericalFailure {
                reason: format!(
                    "resolving the posterior average needs {nodes} parameter states, about {:.1} GiB, above this machine's {:.1} GiB materialisation budget",
                    bytes / f64::from(1u32 << 30),
                    budget / f64::from(1u32 << 30)
                ),
            });
        }
        Ok(())
    }

    /// Floats one parameter state holds: its coefficients, loadings, rates and
    /// reference normaliser.
    fn state_floats(&self) -> usize {
        let marks = self.marks();
        let atoms = self.posterior.atoms;
        let normaliser = self
            .reference
            .as_ref()
            .map_or(0, |law| law.profiles.nrows() * law.grid.len() * marks);
        self.posterior.mode.len() + marks * atoms + atoms + normaliser
    }

    fn materialisation_budget() -> f64 {
        gam_runtime::resource::ResourcePolicy::default_library().max_single_materialization_bytes
            as f64
    }

    /// Evaluate every node of `needed` that has no value yet.
    fn evaluate_points<F>(
        &self,
        needed: &BTreeSet<PointKey>,
        rules: &[NormalRule],
        evaluate: &F,
        values: &mut BTreeMap<PointKey, Evaluated>,
    ) -> Result<(), EventHistoryError>
    where
        F: Fn(&ParameterState) -> Result<Evaluated, EventHistoryError> + Sync,
    {
        let missing: Vec<&PointKey> = needed
            .iter()
            .filter(|key| !values.contains_key(*key))
            .collect();
        let evaluated: Vec<Result<Evaluated, EventHistoryError>> = missing
            .par_iter()
            .map(|key| {
                let state = self.state(key, rules)?;
                evaluate(&*state)
            })
            .collect();
        let mut width = values.values().next().map(|point| point.values.len());
        for (key, result) in missing.into_iter().zip(evaluated) {
            let point = result?;
            let misshapen = width.is_some_and(|w| w != point.values.len())
                || point.errors.len() != point.values.len();
            let invalid = point.values.iter().any(|x| !x.is_finite())
                || point.errors.iter().any(|x| !(x.is_finite() && *x >= 0.0));
            if misshapen || invalid {
                return Err(EventHistoryError::NumericalFailure {
                    reason: "a posterior parameter state returned a non-finite or misshapen forecast or error"
                        .to_string(),
                });
            }
            width = Some(point.values.len());
            values.insert(key.clone(), point);
        }
        Ok(())
    }

    /// The average of `evaluate` over the posterior of every coefficient, by
    /// the dimension-adaptive sparse rule, with the rule's resolution.
    pub(crate) fn posterior_average<F>(&self, evaluate: F) -> Result<Averaged, EventHistoryError>
    where
        F: Fn(&ParameterState) -> Result<Evaluated, EventHistoryError> + Sync,
    {
        let directions = self.posterior.directions.len();
        let mut rules = vec![normal_rule(1)?];
        let mut values: BTreeMap<PointKey, Evaluated> = BTreeMap::new();
        let mode: Index = Vec::new();
        let center = difference_points(&mode, &rules);
        self.evaluate_points(&center.keys().cloned().collect(), &rules, &evaluate, &mut values)?;
        let width = values.values().next().map_or(0, |point| point.values.len());
        let mut settled = combine(&center, &values, None, width)?;
        let mut old: BTreeSet<Index> = BTreeSet::from([mode]);
        let mut active: BTreeMap<Index, Moments> = BTreeMap::new();
        let mut candidates: Vec<Index> = (0..directions).map(|k| vec![(k, 2)]).collect();
        loop {
            let deepest = candidates
                .iter()
                .flat_map(|index| index.iter().map(|entry| entry.1))
                .max()
                .unwrap_or(1);
            while rules.len() < deepest {
                rules.push(normal_rule(rules.len() + 1)?);
            }
            let mut needed = BTreeSet::new();
            let mut points = Vec::with_capacity(candidates.len());
            for index in &candidates {
                let rule = difference_points(index, &rules);
                needed.extend(rule.keys().cloned());
                points.push(rule);
            }
            let fresh = needed.iter().filter(|key| !values.contains_key(*key)).count();
            self.check_budget(values.len() + fresh, width)?;
            self.evaluate_points(&needed, &rules, &evaluate, &mut values)?;
            let anchor = values.get(&PointKey::new());
            for (index, rule) in candidates.drain(..).zip(points) {
                active.insert(index, combine(&rule, &values, anchor, width)?);
            }
            let mut total = settled.clone();
            let mut error = vec![0.0; width];
            for moments in active.values() {
                total.add(moments);
                for (e, contribution) in moments.first.iter().enumerate() {
                    error[e] += contribution.abs();
                }
            }
            // The frontier is resolved once it no longer dominates the error
            // the states' own integrations measured, or the rounding band of
            // the combination where that is larger. The combination forms
            // `terms` rounded products `W_p·d_p` of a weight and a departure
            // from the mode, itself one rounded subtraction, and sums them. By
            // the backward-error model of an inner product it is resolved from
            // zero only beyond `γ_{terms+1}·Σ_p |W_p·d_p|` (`accumulation_band`),
            // with `γ_n = n·u/(1 − n·u)`: that is the rounding floor.
            let bands: Vec<f64> = (0..width)
                .map(|e| accumulation_band(total.terms + 1, total.absolute[e]))
                .collect();
            let thresholds: Vec<f64> = (0..width)
                .map(|e| total.integration[e].max(bands[e]))
                .collect();
            if error.iter().zip(&thresholds).all(|(e, threshold)| e <= threshold) {
                return Ok(Averaged {
                    errors: (0..width)
                        .map(|e| error[e] + total.integration[e] + bands[e])
                        .collect(),
                    values: total.first,
                    evaluations: values.len(),
                });
            }
            // Refine the frontier index whose contribution is largest against
            // the thresholds it has to meet.
            let mut chosen: Option<(f64, Index)> = None;
            for (index, moments) in &active {
                let ratio = moments
                    .first
                    .iter()
                    .zip(&thresholds)
                    .map(|(contribution, threshold)| {
                        if *threshold > 0.0 {
                            contribution.abs() / threshold
                        } else if *contribution != 0.0 {
                            f64::INFINITY
                        } else {
                            0.0
                        }
                    })
                    .fold(0.0, f64::max);
                if chosen.as_ref().is_none_or(|best| ratio > best.0) {
                    chosen = Some((ratio, index.clone()));
                }
            }
            let Some((ratio, index)) = chosen else {
                return Err(EventHistoryError::IntegrationResolution {
                    reason: "the posterior rule exceeds its tolerance with no index left to refine"
                        .to_string(),
                });
            };
            if !ratio.is_finite() && ratio != f64::INFINITY {
                return Err(EventHistoryError::NumericalFailure {
                    reason: "the posterior rule formed a non-finite contribution".to_string(),
                });
            }
            if let Some(moments) = active.remove(&index) {
                settled.add(&moments);
            }
            old.insert(index.clone());
            for direction in 0..directions {
                let candidate = raised(&index, direction);
                if old.contains(&candidate) || active.contains_key(&candidate) {
                    continue;
                }
                if candidate
                    .iter()
                    .all(|entry| old.contains(&lowered(&candidate, entry.0)))
                {
                    candidates.push(candidate);
                }
            }
        }
    }
}

impl ParameterState {
    /// `log M_d(t)` for one stratum at node times, laid out `n * marks + d`,
    /// by linear interpolation in the log of the normaliser on the reference
    /// grid; `None` under the stationary prior's centring.
    pub(crate) fn log_normaliser_at(
        &self,
        predictor: &EventHistoryPredictor,
        stratum: usize,
        times: &[f64],
    ) -> Result<Option<Vec<f64>>, EventHistoryError> {
        let marks = predictor.marks();
        let (Some(law), Some(held)) = (predictor.reference.as_ref(), self.log_normaliser.as_ref())
        else {
            if stratum != 0 {
                return Err(EventHistoryError::InvalidInput {
                    reason: "a model without reference strata requires stratum zero".to_string(),
                });
            }
            return Ok(None);
        };
        let nodes = law.grid.len();
        if stratum >= law.profiles.nrows() {
            return Err(EventHistoryError::InvalidInput {
                reason: format!(
                    "reference stratum {stratum} is outside 0..{}",
                    law.profiles.nrows()
                ),
            });
        }
        let base = stratum * nodes;
        let mut out = Vec::with_capacity(times.len() * marks);
        for &t in times {
            let (lower, weight) = law.grid.locate(t)?;
            for d in 0..marks {
                let low = held[(base + lower) * marks + d];
                let high = held[(base + lower + 1) * marks + d];
                out.push(low + weight * (high - low));
            }
        }
        Ok(Some(out))
    }
}

/// The resolved eigen-directions of a posterior covariance: every eigenvalue
/// above the symmetric eigensolver's rounding band, as its standard deviation
/// and unit vector. An eigenvalue inside the band is not resolved from zero; one
/// negative beyond it is not a covariance.
fn posterior_directions(
    covariance: &Array2<f64>,
) -> Result<Vec<(f64, Vec<f64>)>, EventHistoryError> {
    if covariance.nrows() != covariance.ncols() || covariance.iter().any(|x| !x.is_finite()) {
        return Err(EventHistoryError::InvalidInput {
            reason: "the posterior covariance must be a finite square matrix".to_string(),
        });
    }
    let (values, vectors) = super::covariance::eigenmodes(covariance)?;
    let values = values.to_vec();
    let band = symmetric_spectrum_rounding_band(&values);
    let mut directions = Vec::with_capacity(values.len());
    for (j, &value) in values.iter().enumerate() {
        if value < -band {
            return Err(EventHistoryError::NumericalFailure {
                reason: format!(
                    "the posterior covariance has eigenvalue {value:e}, negative beyond its rounding band {band:e}"
                ),
            });
        }
        if value > band {
            directions.push((value.sqrt(), vectors.column(j).to_vec()));
        }
    }
    Ok(directions)
}

/// Node log-intensities `η⁰` (design × coefficients + offset) of every mark on
/// a row matrix (covariate columns then time), index `row * marks + d`.
pub(crate) fn design_eta0(
    specs: &[TermCollectionSpec],
    betas: &[&Array1<f64>],
    rows: ArrayView2<'_, f64>,
) -> Result<Vec<f64>, EventHistoryError> {
    let marks = betas.len();
    let total = rows.nrows();
    let mut eta0 = vec![0.0; total * marks];
    for (d, beta) in betas.iter().enumerate() {
        let design = build_term_collection_design(rows, &specs[d]).map_err(|error| {
            EventHistoryError::Fit {
                reason: format!("prediction design for mark {d}: {error}"),
            }
        })?;
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

/// The standard-normal Gauss-Hermite rule of a level: `2·level − 1` nodes
/// `√2·x` with weights `w/√π`, symmetrised about the middle node, which is the
/// mode exactly.
struct NormalRule {
    nodes: Vec<f64>,
    weights: Vec<f64>,
}

fn normal_rule(level: usize) -> Result<NormalRule, EventHistoryError> {
    let count = 2 * level - 1;
    let rule = gam_math::quadrature::gauss_hermite_rule(count).map_err(|error| {
        EventHistoryError::IntegrationResolution {
            reason: format!("the posterior rule of {count} nodes: {error}"),
        }
    })?;
    let sqrt_pi = std::f64::consts::PI.sqrt();
    let mirror = |i: usize| count - 1 - i;
    Ok(NormalRule {
        nodes: (0..count)
            .map(|i| std::f64::consts::SQRT_2 * 0.5 * (rule.nodes[i] - rule.nodes[mirror(i)]))
            .collect(),
        weights: (0..count)
            .map(|i| 0.5 * (rule.weights[i] + rule.weights[mirror(i)]) / sqrt_pi)
            .collect(),
    })
}

/// The nodes and signed weights of one multi-index's difference rule
/// `⊗_k (Q_{l_k} − Q_{l_k − 1})`, merged by node. `Q_1` is the mode alone.
fn difference_points(index: &Index, rules: &[NormalRule]) -> BTreeMap<PointKey, f64> {
    let mut points: BTreeMap<PointKey, f64> = BTreeMap::from([(Vec::new(), 1.0)]);
    for &(direction, level) in index {
        let mut next: BTreeMap<PointKey, f64> = BTreeMap::new();
        for (key, weight) in &points {
            for (rule_level, sign) in [(level, 1.0), (level - 1, -1.0)] {
                let rule = &rules[rule_level - 1];
                let middle = rule.nodes.len() / 2;
                for (node, &w) in rule.weights.iter().enumerate() {
                    let mut point = key.clone();
                    if node != middle {
                        point.push((direction, rule_level, node));
                    }
                    *next.entry(point).or_insert(0.0) += sign * weight * w;
                }
            }
        }
        points = next;
    }
    points
}

/// `index` with `direction` one level higher.
fn raised(index: &Index, direction: usize) -> Index {
    let mut out = index.clone();
    match out.iter().position(|entry| entry.0 >= direction) {
        Some(at) if out[at].0 == direction => out[at].1 += 1,
        Some(at) => out.insert(at, (direction, 2)),
        None => out.push((direction, 2)),
    }
    out
}

/// `index` with `direction` one level lower; a direction at level one leaves
/// the index.
fn lowered(index: &Index, direction: usize) -> Index {
    let mut out = index.clone();
    if let Some(at) = out.iter().position(|entry| entry.0 == direction) {
        if out[at].1 > 2 {
            out[at].1 -= 1;
        } else {
            out.remove(at);
        }
    }
    out
}

/// What a set of rule nodes contributes: the weighted sum of their values, the
/// integration error they measured carried in absolute weight, the absolute
/// sum of the weighted terms, and their count.
#[derive(Clone)]
struct Moments {
    first: Vec<f64>,
    integration: Vec<f64>,
    absolute: Vec<f64>,
    terms: usize,
}

impl Moments {
    fn add(&mut self, other: &Moments) {
        for e in 0..self.first.len() {
            self.first[e] += other.first[e];
            self.integration[e] += other.integration[e];
            self.absolute[e] += other.absolute[e];
        }
        self.terms += other.terms;
    }
}

/// The contribution of `points`. The mode's rule is summed as it stands. A
/// difference rule's weights sum to zero, so it is summed over every node's
/// departure from the mode's value `anchor`: a constant forecast then
/// contributes exactly nothing, as it does in exact arithmetic, and the rule
/// reproduces it exactly. The rounded weights do not sum to zero exactly, so
/// the mode's own error enters through their computed sum.
fn combine(
    points: &BTreeMap<PointKey, f64>,
    values: &BTreeMap<PointKey, Evaluated>,
    anchor: Option<&Evaluated>,
    width: usize,
) -> Result<Moments, EventHistoryError> {
    let mut moments = Moments {
        first: vec![0.0; width],
        integration: vec![0.0; width],
        absolute: vec![0.0; width],
        terms: 0,
    };
    let mut weight_sum = 0.0;
    for (key, &weight) in points {
        let point = values.get(key).ok_or_else(|| EventHistoryError::NumericalFailure {
            reason: "a posterior node was combined before it was evaluated".to_string(),
        })?;
        for e in 0..width {
            let departure = anchor.map_or(point.values[e], |mode| point.values[e] - mode.values[e]);
            let term = weight * departure;
            moments.first[e] += term;
            moments.integration[e] += weight.abs() * point.errors[e];
            moments.absolute[e] += term.abs();
        }
        weight_sum += weight;
        moments.terms += 1;
    }
    if let Some(mode) = anchor {
        for e in 0..width {
            moments.integration[e] += weight_sum.abs() * mode.errors[e];
        }
    }
    Ok(moments)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forecast::{
        FutureSegment, PopulationForecastRequest, population_forecast, population_window,
    };

    impl EventHistoryPredictor {
        /// The predictor of a fit conditional on its posterior mode: every
        /// forecast is the plug-in at the fitted coefficients, loadings and
        /// rates, with every rate held. Only a test whose oracle is a closed
        /// form in the fitted parameters reads it. No published forecast does.
        pub(crate) fn conditional_on_mode(
            fit: &EventHistoryFit,
            cohort: &EventHistoryCohort,
        ) -> Result<Self, EventHistoryError> {
            let marks = fit.marks();
            let atoms = fit.rank();
            let mut mode = Vec::new();
            let mut widths = Vec::with_capacity(marks);
            for d in 0..marks {
                let beta = fit.mark_coefficients(d);
                widths.push(beta.len());
                mode.extend(beta.iter().copied());
            }
            for d in 0..marks {
                for k in 0..atoms {
                    mode.push(fit.loadings[[d, k]]);
                }
            }
            let held_rates = fit.log_rates.iter().map(|r| Some(r.exp())).collect();
            Self::on_posterior(fit, cohort, mode, widths, Vec::new(), held_rates)
        }
    }

    /// A predictor whose every mark has an intercept alone, no covariates, on
    /// the given posterior.
    fn intercept_predictor(
        kinds: Vec<MarkKind>,
        mode: Vec<f64>,
        covariance: Array2<f64>,
        held_rates: Vec<Option<f64>>,
        reference: Option<(ReferenceGrid, Array2<f64>)>,
    ) -> EventHistoryPredictor {
        let marks = kinds.len();
        let intercept = TermCollectionSpec {
            linear_terms: Vec::new(),
            random_effect_terms: Vec::new(),
            smooth_terms: Vec::new(),
        };
        EventHistoryPredictor::assemble(PredictorParts {
            mark_names: (0..marks).map(|d| format!("mark{d}")).collect(),
            mark_kinds: kinds,
            covariate_names: Vec::new(),
            covariate_levels: Vec::new(),
            frozen_specs: vec![intercept; marks],
            quadrature_order: crate::cohort::quadrature_order_for_degree(3),
            mesh_refinement: 2,
            time_scale: 1.0,
            gauss_hermite_order: 9,
            mode,
            widths: vec![1; marks],
            directions: posterior_directions(&covariance).expect("posterior directions"),
            atoms: held_rates.len(),
            rate_band: (0.0, 1.0),
            held_rates,
            reference,
        })
        .expect("predictor")
    }

    /// One terminal mark whose log rate has the posterior `N(ln 1.5, 1/4)`. The
    /// survival to 2 is `E[exp(−2·e^β)]`, which the plug-in `exp(−2·1.5)`
    /// misses by the convexity gap. Both are computed independently here: the
    /// average by the trapezoid rule over the standard normal, which converges
    /// geometrically for this integrand.
    #[test]
    fn a_forecast_averages_the_final_probability_over_the_posterior_2964() {
        let mode = 1.5_f64.ln();
        let variance = 0.25;
        let horizon = 2.0;
        let run = |predictor: &EventHistoryPredictor| {
            population_forecast(
                predictor,
                &PopulationForecastRequest {
                    start: 0.0,
                    stratum: 0,
                    horizons: &[horizon],
                    future: &[FutureSegment {
                        start: 0.0,
                        covariates: Vec::new(),
                    }],
                },
            )
            .expect("population forecast")
        };
        let averaged = run(&intercept_predictor(
            vec![MarkKind::Terminal],
            vec![mode],
            Array2::from_elem((1, 1), variance),
            Vec::new(),
            None,
        ));
        let conditional = run(&intercept_predictor(
            vec![MarkKind::Terminal],
            vec![mode],
            Array2::zeros((1, 1)),
            Vec::new(),
            None,
        ));
        // The trapezoid rule on [−12, 12] at 24000 and at 12000 intervals. The
        // difference bounds the finer rule's error, which converges
        // geometrically for this integrand, and the Mills bound φ(12)/12 per
        // side covers the truncated tails, where the integrand is at most one.
        let trapezoid = |intervals: usize| -> f64 {
            let step = 24.0 / intervals as f64;
            (0..=intervals)
                .map(|i| {
                    let z = -12.0 + step * i as f64;
                    let end = if i == 0 || i == intervals { 0.5 } else { 1.0 };
                    let density = (-0.5 * z * z).exp() / (2.0 * std::f64::consts::PI).sqrt();
                    end * step * density * (-horizon * (mode + variance.sqrt() * z).exp()).exp()
                })
                .sum()
        };
        let mean = trapezoid(24000);
        let tails = 2.0 * (-0.5 * 12.0_f64 * 12.0).exp() / (12.0 * (2.0 * std::f64::consts::PI).sqrt());
        let oracle_error = (mean - trapezoid(12000)).abs() + tails;
        let hazard = horizon * mode.exp();
        let plug_in = (-hazard).exp();
        // The closed form's rounding is `ε · μ`, with `μ` the running bound of
        // its chain: the hazard and its exponential. Exponentiating the mode is
        // an operation both routes perform identically, so it adds nothing.
        let plug_in_rounding = f64::EPSILON * (hazard.abs() + plug_in.abs());
        // Positive control: a posterior with no spread is the plug-in, to the
        // window's own checked error and the closed form's rounding. The plug-in
        // misses the average by more than every measured error.
        assert_eq!(conditional.posterior_evaluations, 1);
        assert!(
            (conditional.survival[0] - plug_in).abs()
                <= conditional.survival_error[0] + plug_in_rounding,
            "the mode's survival {} against exp(−3) {plug_in} at checked error {} and oracle rounding {plug_in_rounding}",
            conditional.survival[0],
            conditional.survival_error[0]
        );
        let gap = mean - plug_in;
        let error = (averaged.survival[0] - mean).abs();
        assert!(averaged.posterior_evaluations > 1);
        assert!(
            error <= averaged.survival_error[0] + oracle_error,
            "survival {} against the average {mean}: error {error} above the checked error {} plus the oracle's {oracle_error}",
            averaged.survival[0],
            averaged.survival_error[0]
        );
        assert!(
            gap > averaged.survival_error[0]
                + conditional.survival_error[0]
                + oracle_error
                + plug_in_rounding,
            "the lognormal average {mean} must exceed the plug-in {plug_in} by more than the measured errors, got {gap}"
        );
    }

    /// A forecast that is the same under every parameter state is reproduced
    /// exactly. Recurrent marks kill nothing, so every state's survival is one.
    /// Each difference rule's weights sum to zero, and summed over departures
    /// from the mode they add exactly nothing to it.
    #[test]
    fn a_forecast_the_same_under_every_state_is_reproduced_exactly_2964() {
        let mut covariance = Array2::zeros((2, 2));
        covariance[[0, 0]] = 0.25;
        covariance[[1, 1]] = 0.16;
        let predictor = intercept_predictor(
            vec![MarkKind::Recurrent, MarkKind::Recurrent],
            vec![0.5_f64.ln(), 0.8_f64.ln()],
            covariance,
            Vec::new(),
            None,
        );
        let forecast = population_forecast(
            &predictor,
            &PopulationForecastRequest {
                start: 0.0,
                stratum: 0,
                horizons: &[1.0, 2.0],
                future: &[FutureSegment {
                    start: 0.0,
                    covariates: Vec::new(),
                }],
            },
        )
        .expect("population forecast");
        assert!(
            forecast.posterior_evaluations > 1,
            "the rule evaluated the mode alone"
        );
        assert_eq!(
            forecast.survival,
            vec![1.0, 1.0],
            "a survival of one under every state is published as exactly one"
        );
    }

    /// One once-only mark at rank one with a held rate, centred on a reference
    /// population of one stratum over `[0, 4]`, with an uncertain baseline and
    /// loading: every part a predictor holds.
    ///
    /// Every state the rule visits evolves its own reference law, whose midpoint
    /// step is refused where it does not contract. At step 1/2 with a log
    /// baseline of zero, a loading of 2 contracts
    /// (`reference_midpoint_refuses_only_a_step_that_does_not_contract_2627`).
    /// The spreads here keep every node out to 11 standard deviations, level 30
    /// of the rule, at a loading below 2 and a log baseline below zero. A loading
    /// spread of 0.4 reached a non-contracting step in job 1212458.
    fn reference_predictor() -> EventHistoryPredictor {
        let times: Vec<f64> = (0..=8).map(|n| 0.5 * n as f64).collect();
        let grid = ReferenceGrid {
            gaps: times.windows(2).map(|w| w[1] - w[0]).collect(),
            times,
        };
        let mut covariance = Array2::zeros((2, 2));
        covariance[[0, 0]] = 0.01;
        covariance[[1, 1]] = 0.0081;
        intercept_predictor(
            vec![MarkKind::Once],
            vec![0.2_f64.ln(), 0.9],
            covariance,
            vec![Some(0.3)],
            Some((grid, Array2::zeros((1, 0)))),
        )
    }

    /// Under reference centring a state's normaliser moves with its own
    /// baseline and loading. The published forecast divides every state by its
    /// own law. Dividing every state by the mode's law instead gives a forecast
    /// that differs by more than both resolutions.
    #[test]
    fn every_posterior_state_divides_by_its_own_reference_law_2964() {
        let predictor = reference_predictor();
        let request = PopulationForecastRequest {
            start: 1.0,
            stratum: 0,
            horizons: &[2.0, 3.5],
            future: &[FutureSegment {
                start: 1.0,
                covariates: Vec::new(),
            }],
        };
        let published = population_forecast(&predictor, &request).expect("forecast");
        let rules = [normal_rule(1).expect("rule"), normal_rule(2).expect("rule")];
        let mode = predictor.state(&Vec::new(), &rules).expect("mode state");
        // The law does move with the state: off the mode along either
        // direction the normaliser differs.
        for direction in 0..2 {
            let off = predictor
                .state(&vec![(direction, 2, 0)], &rules)
                .expect("state off the mode");
            // Two evaluations of the evolution agree to the rounding band of
            // their summands; a law that moves has to leave it.
            let (moved, band) = off
                .log_normaliser
                .as_ref()
                .zip(mode.log_normaliser.as_ref())
                .map(|(a, b)| {
                    let moved = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max);
                    let scale: f64 = a.iter().chain(b).map(|x| x.abs()).sum();
                    (moved, accumulation_band(a.len() + b.len(), scale))
                })
                .expect("reference normalisers");
            assert!(
                moved > band,
                "direction {direction}: the normaliser moved by {moved}, inside its rounding band {band}"
            );
        }
        let own = predictor
            .posterior_average(|state| population_window(&predictor, state, &request))
            .expect("average under each state's own law");
        let borrowed = predictor
            .posterior_average(|state| {
                let divided_by_the_mode = ParameterState {
                    betas: state.betas.clone(),
                    loadings: state.loadings.clone(),
                    rates: state.rates.clone(),
                    log_normaliser: mode.log_normaliser.clone(),
                };
                population_window(&predictor, &divided_by_the_mode, &request)
            })
            .expect("average under the mode's law");
        // Layout: survival per horizon, then the counts row-major.
        let last = request.horizons.len() + request.horizons.len() - 1;
        assert_eq!(published.expected_counts[[1, 0]], own.values[last]);
        assert_eq!(published.expected_count_errors[[1, 0]], own.errors[last]);
        let shift = (own.values[last] - borrowed.values[last]).abs();
        assert!(
            shift > own.errors[last] + borrowed.errors[last],
            "own law {} against the mode's law {}: shift {shift}, checked errors {} and {}",
            own.values[last],
            borrowed.values[last],
            own.errors[last],
            borrowed.errors[last]
        );
        let checked = published.expected_count_errors[[1, 0]];
        assert!(
            published.expected_counts[[1, 0]] > -checked
                && published.expected_counts[[1, 0]] < 1.0 + checked,
            "a first-occurrence probability {} at checked error {checked}",
            published.expected_counts[[1, 0]]
        );
    }

    /// The admission check reads the machine's own materialisation budget. It
    /// admits the largest rule whose states and forecasts fit that budget, and
    /// refuses one more node with a typed error, before anything is evaluated.
    /// Both counts are derived from the budget the production check reads and
    /// from what one node holds.
    #[test]
    fn a_posterior_rule_past_the_materialisation_budget_is_refused_before_it_grows_2964() {
        let predictor = reference_predictor();
        let horizons = 2;
        let width = horizons * (1 + predictor.marks());
        let marks = predictor.marks();
        let atoms = predictor.atoms();
        let law = predictor.reference.as_ref().expect("reference law");
        let per_node = predictor.posterior.mode.len()
            + marks * atoms
            + atoms
            + law.profiles.nrows() * law.grid.len() * marks
            + 2 * width;
        let budget = gam_runtime::resource::ResourcePolicy::default_library()
            .max_single_materialization_bytes as f64;
        let admitted = (budget / (per_node * std::mem::size_of::<f64>()) as f64).floor() as usize;
        assert!(
            predictor.check_budget(admitted, width).is_ok(),
            "{admitted} nodes of {per_node} floats fit a budget of {budget} bytes"
        );
        match predictor.check_budget(admitted + 1, width) {
            Err(EventHistoryError::NumericalFailure { reason }) => assert!(
                reason.contains("materialisation budget"),
                "the refusal names the budget: {reason}"
            ),
            other => panic!("one node past the budget must be refused, got {other:?}"),
        }
        // Admission counts this request's nodes only. A cache filled by an
        // earlier forecast changes nothing, so the same request is admitted
        // whatever ran before it.
        let request = PopulationForecastRequest {
            start: 1.0,
            stratum: 0,
            horizons: &[2.0, 3.5],
            future: &[FutureSegment {
                start: 1.0,
                covariates: Vec::new(),
            }],
        };
        population_forecast(&predictor, &request).expect("an earlier forecast");
        let cached = predictor.states.lock().map_or(0, |cache| cache.len());
        assert!(cached > 0, "the earlier forecast left no parameter state cached");
        assert!(
            predictor.check_budget(admitted, width).is_ok(),
            "{admitted} nodes no longer fit after an earlier forecast cached {cached} states"
        );
    }
}
