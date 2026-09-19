//! Python bindings for the event-history family: fit from arrays, then
//! forecast and score subjects from the in-memory fit. Every rule about the
//! data (mark kinds, categorical levels, follow-up) lives in the Rust cohort
//! validation; this layer only moves arrays.

use crate::ffi::ffi_errors::{detach_py_result, py_value_error};
use gam::families::custom_family::BlockwiseFitOptions;
use gam::event_history::{
    CovariateCells, CovariateSegment, CovariateValue, Event, EventHistoryCohort, EventHistoryFit,
    ForecastRequest, FutureSegment, HistoryForecastRequest, MarkKind, PopulationForecastRequest,
    ReferenceStrata, SubjectHistory, code_covariate_value, fit_event_history_formulas, forecast,
    forecast_history, latent_state, mark_index_of, pit_uniform_distance, population_forecast,
    predictive_pit, resolve_mark_vocabulary,
};
use ndarray::{Array2, Array3};
use numpy::{PyArray1, PyArray2, PyArray3};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyModule};
use std::sync::Arc;

/// A fitted event-history model held in memory.
#[pyclass(name = "_EventHistoryModel", frozen)]
pub(crate) struct PyEventHistoryModel {
    fit: Arc<EventHistoryFit>,
    cohort: Arc<EventHistoryCohort>,
    /// Each training subject's reference stratum, so prediction divides by
    /// the normaliser of the population that subject belongs to.
    strata: Vec<usize>,
}

/// Code one Python covariate record against the fitted schema: a continuous
/// covariate takes a number, a categorical one the `str` of its value, and the
/// cohort's levels decide the code.
fn coded_record(cohort: &EventHistoryCohort, values: &[Bound<'_, PyAny>]) -> PyResult<Vec<f64>> {
    if values.len() != cohort.covariate_names.len() {
        return Err(py_value_error(format!(
            "a covariate record has {} values for {} covariates",
            values.len(),
            cohort.covariate_names.len()
        )));
    }
    cohort
        .covariate_names
        .iter()
        .zip(&cohort.covariate_levels)
        .zip(values)
        .map(|((name, levels), value)| {
            let value = if levels.is_empty() {
                CovariateValue::Number(value.extract::<f64>()?)
            } else {
                CovariateValue::Label(value.str()?.to_string())
            };
            code_covariate_value(name, levels, value).map_err(|e| py_value_error(e.to_string()))
        })
        .collect()
}

fn future_segments(
    cohort: &EventHistoryCohort,
    future: Vec<(f64, Vec<Bound<'_, PyAny>>)>,
) -> PyResult<Vec<FutureSegment>> {
    future
        .into_iter()
        .map(|(start, record)| {
            Ok(FutureSegment {
                start,
                covariates: coded_record(cohort, &record)?,
            })
        })
        .collect()
}

fn forecast_dict<'py>(
    py: Python<'py>,
    result: gam::event_history::Forecast,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("horizons", result.horizons)?;
    out.set_item("survival", result.survival)?;
    out.set_item(
        "expected_counts",
        PyArray2::from_owned_array(py, result.expected_counts),
    )?;
    Ok(out)
}

#[pymethods]
impl PyEventHistoryModel {
    fn mark_names(&self) -> Vec<String> {
        self.cohort.mark_names.clone()
    }

    fn mark_kinds(&self) -> Vec<String> {
        self.cohort
            .mark_kinds
            .iter()
            .map(|k| k.name().to_string())
            .collect()
    }

    fn covariate_names(&self) -> Vec<String> {
        self.cohort.covariate_names.clone()
    }

    fn covariate_levels(&self) -> Vec<Vec<String>> {
        self.cohort.covariate_levels.clone()
    }

    fn subject_ids(&self) -> Vec<String> {
        self.cohort.subjects.iter().map(|s| s.id.clone()).collect()
    }

    fn subject_exits(&self) -> Vec<f64> {
        self.cohort.subjects.iter().map(|s| s.exit).collect()
    }

    fn rank(&self) -> usize {
        self.fit.rank()
    }

    /// Per reference grid the fit ran on, the move the next grid makes at the
    /// fitted coefficients, in posterior standard deviations; empty for
    /// stationary-prior centring.
    fn reference_refinements(&self) -> Vec<f64> {
        self.fit.reference_refinements.clone()
    }

    fn reference_masks(&self) -> usize {
        self.fit.centring.as_ref().map_or(0, |c| c.masks)
    }

    /// The reference grid's certificate: the geometric-tail estimate of the
    /// moves finer grids make the fitted coefficients take, in posterior
    /// standard deviations.
    fn reference_certificate(&self) -> Option<f64> {
        self.fit.reference_certificate
    }

    fn atom_evidence(&self) -> Vec<f64> {
        self.fit.atom_evidence.clone()
    }

    fn rank_path<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let out = PyList::empty(py);
        for step in &self.fit.rank_path {
            let item = PyDict::new(py);
            item.set_item("rank", step.rank)?;
            item.set_item("score_eigenvalue", step.score_eigenvalue)?;
            item.set_item("standardised_gain", step.standardised_gain)?;
            item.set_item("proposed_rate", step.proposed_rate)?;
            item.set_item("at_resolution_limit", step.at_resolution_limit)?;
            item.set_item("rate_held", step.rate_held)?;
            item.set_item("ridge_log_lambda", step.ridge_log_lambda)?;
            item.set_item("evidence_gain", step.evidence_gain)?;
            item.set_item("log_likelihood_gain", step.log_likelihood_gain)?;
            item.set_item("accepted", step.accepted)?;
            item.set_item("converged", step.converged)?;
            let growth_unresolved = match &step.growth_unresolved {
                Some(growth) => {
                    let entry = PyDict::new(py);
                    entry.set_item("gauss_hermite_order", growth.gauss_hermite_order)?;
                    entry.set_item("integral", growth.integral.name())?;
                    entry.set_item("reason", &growth.reason)?;
                    Some(entry)
                }
                None => None,
            };
            item.set_item("growth_unresolved", growth_unresolved)?;
            out.append(item)?;
        }
        Ok(out)
    }

    fn covariance<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        PyArray2::from_owned_array(py, self.fit.covariance.clone())
    }

    fn temporal_covariance<'py>(&self, py: Python<'py>, lag: f64) -> Bound<'py, PyArray2<f64>> {
        PyArray2::from_owned_array(py, self.fit.temporal_covariance(lag))
    }

    fn eigenvalues<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_owned_array(py, self.fit.eigenvalues.clone())
    }

    fn eigenvalue_sd<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_owned_array(py, self.fit.eigenvalue_sd.clone())
    }

    fn eigenvectors<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        PyArray2::from_owned_array(py, self.fit.eigenvectors.clone())
    }

    fn effective_rank(&self) -> f64 {
        self.fit.effective_rank
    }

    fn loadings<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        PyArray2::from_owned_array(py, self.fit.loadings.clone())
    }

    /// The smoothed latent state of one training subject: node times, the
    /// posterior mean (`nodes × atoms`) and the posterior covariance
    /// (`nodes × atoms × atoms`) of the atoms given the whole history.
    fn latent_state<'py>(&self, py: Python<'py>, subject: usize) -> PyResult<Bound<'py, PyDict>> {
        let stratum = self.strata.get(subject).copied().unwrap_or(0);
        let history = self
            .cohort
            .subjects
            .get(subject)
            .ok_or_else(|| py_value_error(format!("subject index {subject} is out of range")))?
            .clone();
        let fit = Arc::clone(&self.fit);
        let cohort = Arc::clone(&self.cohort);
        let state = detach_py_result(py, "event-history latent state", move || {
            latent_state(&fit, &cohort, &history, stratum).map_err(|e| e.to_string())
        })?;
        let atoms = self.fit.rank();
        let mut covariance = Array3::<f64>::zeros((state.times.len(), atoms, atoms));
        for (n, matrix) in state.covariance.iter().enumerate() {
            covariance
                .index_axis_mut(ndarray::Axis(0), n)
                .assign(matrix);
        }
        let out = PyDict::new(py);
        out.set_item("time", state.times)?;
        out.set_item("mean", PyArray2::from_owned_array(py, state.mean))?;
        out.set_item("covariance", PyArray3::from_owned_array(py, covariance))?;
        Ok(out)
    }

    fn rates(&self) -> Vec<f64> {
        self.fit.rates.clone()
    }

    fn rate_held(&self) -> Vec<bool> {
        self.fit.rate_held.clone()
    }

    fn atom_log_lambdas(&self) -> Vec<f64> {
        self.fit.atom_log_lambdas.clone()
    }

    fn time_scale(&self) -> f64 {
        self.fit.time_scale
    }

    fn log_likelihood(&self) -> f64 {
        self.fit.fit.log_likelihood
    }

    fn reml_score(&self) -> Option<f64> {
        self.fit.fit.reml_score()
    }

    fn outer_iterations(&self) -> usize {
        self.fit.fit.outer_iterations
    }

    fn coefficients(&self, mark: usize) -> PyResult<Vec<f64>> {
        if mark >= self.fit.marks() {
            return Err(py_value_error(format!(
                "mark index {mark} is outside the {} marks of the fit",
                self.fit.marks()
            )));
        }
        Ok(self.fit.mark_coefficients(mark).to_vec())
    }

    fn baseline_rates<'py>(&self, py: Python<'py>, values: Vec<Bound<'py, PyAny>>, times: Vec<f64>, stratum: usize) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let values = coded_record(&self.cohort, &values)?;
        let mut rows = Array2::zeros((times.len(), values.len() + 1));
        for (n, &time) in times.iter().enumerate() {
            self.fit.risk_set_normaliser_at(stratum, time).map_err(|e| py_value_error(e.to_string()))?;
            for (j, &value) in values.iter().enumerate() { rows[[n, j]] = value; }
            rows[[n, values.len()]] = time;
        }
        let rates = gam::event_history::baseline_log_rates(&self.fit, rows.view())
            .map_err(|e| py_value_error(e.to_string()))?.mapv(f64::exp);
        if rates.iter().any(|x| !x.is_finite()) {
            return Err(py_value_error("baseline rate exceeds floating-point range".to_string()));
        }
        Ok(PyArray2::from_owned_array(py, rates))
    }

    fn quadrature<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let q = &self.fit.quadrature;
        let out = PyDict::new(py);
        out.set_item("gauss_hermite_order", q.gauss_hermite_order)?;
        out.set_item("mesh_refinement", q.mesh_refinement)?;
        out.set_item("log_likelihood", q.log_likelihood)?;
        let gh = PyDict::new(py);
        gh.set_item("order", q.gauss_hermite.candidate)?;
        gh.set_item("coefficient_shift", q.gauss_hermite.coefficient_shift)?;
        gh.set_item("log_likelihood", q.gauss_hermite.log_likelihood)?;
        out.set_item("gauss_hermite_check", gh)?;
        let mesh = PyDict::new(py);
        mesh.set_item("refinement", q.mesh.candidate)?;
        mesh.set_item("coefficient_shift", q.mesh.coefficient_shift)?;
        mesh.set_item("log_likelihood", q.mesh.log_likelihood)?;
        out.set_item("mesh_check", mesh)?;
        Ok(out)
    }

    /// Forecast one training subject beyond its exit over a covariate path.
    #[pyo3(signature = (subject, horizons, future))]
    fn forecast<'py>(
        &self,
        py: Python<'py>,
        subject: usize,
        horizons: Vec<f64>,
        future: Vec<(f64, Vec<Bound<'py, PyAny>>)>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let stratum = self.strata.get(subject).copied().unwrap_or(0);
        let history = self
            .cohort
            .subjects
            .get(subject)
            .ok_or_else(|| py_value_error(format!("subject index {subject} is out of range")))?
            .clone();
        let fit = Arc::clone(&self.fit);
        let cohort = Arc::clone(&self.cohort);
        let future = future_segments(&self.cohort, future)?;
        let result = detach_py_result(py, "event-history forecast", move || {
            forecast(
                &fit,
                &cohort,
                &ForecastRequest {
                    history: &history,
                    horizons: &horizons,
                    future: &future,
                    stratum,
                },
            )
            .map_err(|e| e.to_string())
        })?;
        forecast_dict(py, result)
    }

    /// Forecast a subject with no history from a covariate path alone.
    #[pyo3(signature = (start, horizons, future, stratum))]
    fn population_forecast<'py>(
        &self,
        py: Python<'py>,
        start: f64,
        horizons: Vec<f64>,
        future: Vec<(f64, Vec<Bound<'py, PyAny>>)>,
        stratum: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        let fit = Arc::clone(&self.fit);
        let cohort = Arc::clone(&self.cohort);
        let future = future_segments(&self.cohort, future)?;
        let result = detach_py_result(py, "event-history population forecast", move || {
            population_forecast(
                &fit,
                &cohort,
                &PopulationForecastRequest {
                    start,
                    horizons: &horizons,
                    future: &future,
                    stratum,
                },
            )
            .map_err(|e| e.to_string())
        })?;
        forecast_dict(py, result)
    }

    /// Predictive PIT of every spell of one training subject: the end
    /// times, whether each spell ended with an event, the PIT values, the
    /// marks that fired (one list per spell, empty for the censored tail)
    /// and the predictive mark probabilities at each spell's end.
    fn pit<'py>(&self, py: Python<'py>, subject: usize) -> PyResult<Bound<'py, PyDict>> {
        let stratum = self.strata.get(subject).copied().unwrap_or(0);
        let history = self
            .cohort
            .subjects
            .get(subject)
            .ok_or_else(|| py_value_error(format!("subject index {subject} is out of range")))?
            .clone();
        let fit = Arc::clone(&self.fit);
        let cohort = Arc::clone(&self.cohort);
        let pits = detach_py_result(py, "event-history pit", move || {
            predictive_pit(&fit, &cohort, &history, stratum).map_err(|e| e.to_string())
        })?;
        let marks = self.fit.marks();
        let mut probabilities = Array2::<f64>::zeros((pits.len(), marks));
        for (i, pit) in pits.iter().enumerate() {
            for d in 0..marks {
                probabilities[[i, d]] = pit.mark_probabilities[d];
            }
        }
        let out = PyDict::new(py);
        out.set_item("time", pits.iter().map(|p| p.time).collect::<Vec<_>>())?;
        out.set_item(
            "observed",
            pits.iter().map(|p| p.observed).collect::<Vec<_>>(),
        )?;
        out.set_item("pit", pits.iter().map(|p| p.pit).collect::<Vec<_>>())?;
        out.set_item(
            "marks",
            pits.iter().map(|p| p.marks.clone()).collect::<Vec<_>>(),
        )?;
        out.set_item(
            "mark_probabilities",
            PyArray2::from_owned_array(py, probabilities),
        )?;
        Ok(out)
    }

    /// The Kaplan–Meier distance of the cohort's predictive PITs from the
    /// uniform law over event and censored spells, with the spell and event
    /// counts it was read from; the distance is `None` for no spells.
    fn pit_distance<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let fit = Arc::clone(&self.fit);
        let cohort = Arc::clone(&self.cohort);
        let strata = self.strata.clone();
        let (distance, spells, events) = detach_py_result(py, "event-history pit", move || {
            let mut pits = Vec::new();
            for (index, subject) in cohort.subjects.iter().enumerate() {
                pits.extend(
                    predictive_pit(&fit, &cohort, subject, strata[index])
                        .map_err(|e| e.to_string())?,
                );
            }
            let events = pits.iter().filter(|p| p.observed).count();
            Ok((pit_uniform_distance(&pits), pits.len(), events))
        })?;
        let out = PyDict::new(py);
        out.set_item("distance", distance)?;
        out.set_item("spells", spells)?;
        out.set_item("events", events)?;
        Ok(out)
    }

    /// Forecast a history that is not a training subject's — or a training
    /// subject's prefix — from its own records: entry and exit, events as
    /// parallel time and mark-label vectors, covariate segments as parallel
    /// start times and `covariates` records (one per segment, in the cohort's
    /// columns, coded here against its levels). With `cutoff`, the history is cut
    /// to what was known at the cutoff before forecasting.
    #[pyo3(signature = (entry, exit, event_time, event_marks, segment_start, covariates, cutoff, horizons, future, stratum))]
    fn forecast_history<'py>(
        &self,
        py: Python<'py>,
        entry: f64,
        exit: f64,
        event_time: Vec<f64>,
        event_marks: Vec<String>,
        segment_start: Vec<f64>,
        covariates: Vec<Vec<Bound<'py, PyAny>>>,
        cutoff: Option<f64>,
        horizons: Vec<f64>,
        future: Vec<(f64, Vec<Bound<'py, PyAny>>)>,
        stratum: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        if event_time.len() != event_marks.len() {
            return Err(py_value_error(
                "event_time and event_marks must have equal length".to_string(),
            ));
        }
        let event_mark = event_marks
            .iter()
            .map(|label| {
                mark_index_of(&self.cohort.mark_names, label).map_err(|e| py_value_error(e.to_string()))
            })
            .collect::<PyResult<Vec<usize>>>()?;
        let mut table = Array2::<f64>::zeros((covariates.len(), self.cohort.covariate_names.len()));
        for (row, record) in covariates.iter().enumerate() {
            for (j, code) in coded_record(&self.cohort, record)?.into_iter().enumerate() {
                table[[row, j]] = code;
            }
        }
        if segment_start.len() != table.nrows() {
            return Err(py_value_error(format!(
                "{} segment starts for {} covariate rows",
                segment_start.len(),
                table.nrows()
            )));
        }
        let history = SubjectHistory {
            id: "history".to_string(),
            entry,
            exit,
            events: event_time
                .iter()
                .zip(event_mark.iter())
                .map(|(&time, &mark)| Event { time, mark })
                .collect(),
            segments: segment_start
                .iter()
                .enumerate()
                .map(|(row, &start)| CovariateSegment { start, row })
                .collect(),
        };
        let fit = Arc::clone(&self.fit);
        let cohort = Arc::clone(&self.cohort);
        let future = future_segments(&self.cohort, future)?;
        let result = detach_py_result(py, "event-history forecast", move || {
            let history = match cutoff {
                Some(cutoff) => history
                    .prefix(cutoff, &cohort.mark_kinds)
                    .map_err(|e| e.to_string())?,
                None => history,
            };
            forecast_history(
                &fit,
                &cohort,
                &HistoryForecastRequest {
                    history: &history,
                    covariates: table.view(),
                    horizons: &horizons,
                    future: &future,
                    stratum,
                },
            )
            .map_err(|e| e.to_string())
        })?;
        forecast_dict(py, result)
    }
}

/// Fit an event-history model from flat arrays.
#[pyfunction]
#[pyo3(signature = (declared_marks, covariate_names, covariate_columns, subject_ids, entry, exit, event_subject, event_time, event_marks, segment_subject, segment_start, segment_row, formulas, reference_rows, reference_stratum))]
fn fit_event_history(
    py: Python<'_>,
    declared_marks: Option<Vec<(String, String)>>,
    covariate_names: Vec<String>,
    covariate_columns: Vec<Bound<'_, PyAny>>,
    subject_ids: Vec<String>,
    entry: Vec<f64>,
    exit: Vec<f64>,
    event_subject: Vec<usize>,
    event_time: Vec<f64>,
    event_marks: Vec<String>,
    segment_subject: Vec<usize>,
    segment_start: Vec<f64>,
    segment_row: Vec<usize>,
    formulas: Vec<String>,
    reference_rows: Vec<usize>,
    reference_stratum: Vec<usize>,
) -> PyResult<PyEventHistoryModel> {
    let n = subject_ids.len();
    if entry.len() != n || exit.len() != n {
        return Err(py_value_error(format!(
            "subject_ids, entry and exit must have one entry per subject ({n})"
        )));
    }
    if event_subject.len() != event_time.len() || event_subject.len() != event_marks.len() {
        return Err(py_value_error(
            "event_subject, event_time and event_marks must have equal length".to_string(),
        ));
    }
    if segment_subject.len() != segment_start.len() || segment_subject.len() != segment_row.len() {
        return Err(py_value_error(
            "segment_subject, segment_start and segment_row must have equal length".to_string(),
        ));
    }
    let declared = declared_marks
        .map(|pairs| {
            pairs
                .into_iter()
                .map(|(name, kind)| {
                    MarkKind::parse(&kind)
                        .map(|kind| (name, kind))
                        .map_err(|e| py_value_error(e.to_string()))
                })
                .collect::<PyResult<Vec<(String, MarkKind)>>>()
        })
        .transpose()?;
    let event_mark_labels: Vec<&str> = event_marks.iter().map(String::as_str).collect();
    let (mark_names, mark_kinds, event_mark) = resolve_mark_vocabulary(declared, &event_mark_labels)
        .map_err(|e| py_value_error(e.to_string()))?;
    if covariate_columns.len() != covariate_names.len() {
        return Err(py_value_error(format!(
            "{} covariate columns for {} covariate names",
            covariate_columns.len(),
            covariate_names.len()
        )));
    }
    // A column of labels is categorical; any other column is continuous. The
    // shared cohort encoder codes both, as the CLI's does.
    let mut encoded = Vec::with_capacity(covariate_columns.len());
    for column in &covariate_columns {
        let cells = match column.extract::<Vec<String>>() {
            Ok(labels) => CovariateCells::Labels(labels),
            Err(_) => CovariateCells::Numbers(column.extract::<Vec<f64>>()?),
        };
        encoded.push(cells.encode());
    }
    let rows = segment_row.iter().copied().max().map_or(0, |row| row + 1);
    let rows = encoded.first().map_or(rows, |(codes, _)| codes.len());
    let mut covariates = Array2::<f64>::zeros((rows, encoded.len()));
    let mut covariate_levels = Vec::with_capacity(encoded.len());
    for (j, (codes, levels)) in encoded.into_iter().enumerate() {
        if codes.len() != rows {
            return Err(py_value_error(format!(
                "covariate {:?} has {} values for {rows} rows",
                covariate_names[j],
                codes.len()
            )));
        }
        for (i, code) in codes.into_iter().enumerate() {
            covariates[[i, j]] = code;
        }
        covariate_levels.push(levels);
    }
    let mut subjects: Vec<SubjectHistory> = subject_ids
        .iter()
        .zip(entry.iter().zip(exit.iter()))
        .map(|(id, (&entry, &exit))| SubjectHistory {
            id: id.clone(),
            entry,
            exit,
            events: Vec::new(),
            segments: Vec::new(),
        })
        .collect();
    for ((&s, &t), &m) in event_subject
        .iter()
        .zip(event_time.iter())
        .zip(event_mark.iter())
    {
        let subject = subjects
            .get_mut(s)
            .ok_or_else(|| py_value_error(format!("event subject index {s} is out of range")))?;
        subject.events.push(Event { time: t, mark: m });
    }
    for ((&s, &start), &row) in segment_subject
        .iter()
        .zip(segment_start.iter())
        .zip(segment_row.iter())
    {
        let subject = subjects
            .get_mut(s)
            .ok_or_else(|| py_value_error(format!("segment subject index {s} is out of range")))?;
        subject.segments.push(CovariateSegment { start, row });
    }
    let mut cohort = EventHistoryCohort {
        mark_names,
        mark_kinds,
        covariate_names,
        covariate_levels,
        covariates,
        subjects,
    };
    // The reference population, when the baselines are to be the incidence
    // among those still at risk rather than the rate over the cohort as it
    // started. No rows means the stationary prior's centring, which is the
    // model the family has always fitted.
    let subject_strata = if reference_stratum.is_empty() {
        vec![0usize; subject_ids.len()]
    } else {
        reference_stratum.clone()
    };
    let reference = if reference_rows.is_empty() {
        None
    } else {
        Some(ReferenceStrata {
            rows: reference_rows,
            subject: reference_stratum,
        })
    };
    let (fit, cohort) = detach_py_result(py, "event-history fit", move || {
        let fit = fit_event_history_formulas(
            &mut cohort,
            &formulas,
            BlockwiseFitOptions::default(),
            reference,
        )
        .map_err(|e| e.to_string())?;
        Ok((fit, cohort))
    })?;
    let strata = if fit.centring.is_none() {
        vec![0usize; cohort.subjects.len()]
    } else {
        subject_strata
    };
    Ok(PyEventHistoryModel {
        fit: Arc::new(fit),
        cohort: Arc::new(cohort),
        strata,
    })
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyEventHistoryModel>()?;
    module.add_function(wrap_pyfunction!(fit_event_history, module)?)?;
    Ok(())
}
