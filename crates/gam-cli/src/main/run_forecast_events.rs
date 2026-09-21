//! `gam forecast-events` and `gam forecast-population`: posterior-predictive
//! forecasts of new histories, or of a subject with no history from a covariate
//! path alone, from a saved event-history predictor, with no training records.

use crate::cli_args::{ForecastEventsArgs, ForecastPopulationArgs};
use crate::run_fit_events::{column, forecast_json, parse_f64, read_csv, write_json_output};
use gam::event_history::{
    CovariateSegment, CovariateValue, Event, EventHistoryPredictor, FutureSegment,
    HistoryForecastRequest, PopulationForecastRequest, SubjectHistory, code_covariate_value,
    forecast_history, mark_index_of, population_forecast,
};
use ndarray::Array2;
use serde_json::{Map, Value, json};
use std::collections::HashMap;

pub(crate) fn run_forecast_events(args: ForecastEventsArgs) -> Result<(), String> {
    let predictor = EventHistoryPredictor::load(&args.model).map_err(|e| e.to_string())?;
    let (subject_headers, subject_rows) = read_csv(&args.subjects)?;
    let ids = column(&subject_headers, &subject_rows, "id", &args.subjects)?;
    let entries = column(&subject_headers, &subject_rows, "entry", &args.subjects)?;
    let exits = column(&subject_headers, &subject_rows, "exit", &args.subjects)?;
    let strata: Vec<usize> = match &args.reference_stratum {
        None => vec![0; ids.len()],
        Some(name) => column(&subject_headers, &subject_rows, name, &args.subjects)?
            .iter()
            .map(|v| {
                v.parse::<usize>().map_err(|_| {
                    format!(
                        "reference stratum {v:?} in column {name:?} must be a nonnegative integer index"
                    )
                })
            })
            .collect::<Result<Vec<_>, String>>()?,
    };
    let mut index: HashMap<String, usize> = HashMap::new();
    let mut subjects: Vec<SubjectHistory> = Vec::with_capacity(ids.len());
    for (i, ((id, entry), exit)) in ids.iter().zip(entries.iter()).zip(exits.iter()).enumerate() {
        if index.insert((*id).to_string(), i).is_some() {
            return Err(format!(
                "duplicate subject id {id:?} in {}",
                args.subjects.display()
            ));
        }
        subjects.push(SubjectHistory {
            id: (*id).to_string(),
            entry: parse_f64(entry, "entry")?,
            exit: parse_f64(exit, "exit")?,
            events: Vec::new(),
            segments: Vec::new(),
        });
    }
    let (event_headers, event_rows) = read_csv(&args.events)?;
    let event_ids = column(&event_headers, &event_rows, "id", &args.events)?;
    let event_times = column(&event_headers, &event_rows, "time", &args.events)?;
    let event_marks = column(&event_headers, &event_rows, "mark", &args.events)?;
    for ((id, time), mark) in event_ids.iter().zip(event_times.iter()).zip(event_marks.iter()) {
        let subject = *index
            .get(*id)
            .ok_or_else(|| format!("event subject {id:?} is not in the subjects table"))?;
        subjects[subject].events.push(Event {
            time: parse_f64(time, "event time")?,
            mark: mark_index_of(&predictor.mark_names, mark).map_err(|e| e.to_string())?,
        });
    }
    // Every subject's covariate rows are its own table, coded against the
    // saved levels rather than re-encoded from these records.
    let (cov_headers, cov_rows) = read_csv(&args.covariates)?;
    let cov_ids = column(&cov_headers, &cov_rows, "id", &args.covariates)?;
    let cov_starts = column(&cov_headers, &cov_rows, "start", &args.covariates)?;
    let cells = predictor
        .covariate_names
        .iter()
        .map(|name| column(&cov_headers, &cov_rows, name, &args.covariates))
        .collect::<Result<Vec<_>, String>>()?;
    let mut records: Vec<Vec<Vec<f64>>> = vec![Vec::new(); subjects.len()];
    for (row, (id, start)) in cov_ids.iter().zip(cov_starts.iter()).enumerate() {
        let subject = *index
            .get(*id)
            .ok_or_else(|| format!("covariate subject {id:?} is not in the subjects table"))?;
        let record = coded_row(&predictor, &cells, row)?;
        subjects[subject].segments.push(CovariateSegment {
            start: parse_f64(start, "segment start")?,
            row: records[subject].len(),
        });
        records[subject].push(record);
    }
    let mut forecasts = Vec::with_capacity(subjects.len());
    let mut skipped = 0usize;
    for (i, subject) in subjects.iter().enumerate() {
        // With a cutoff, the history is what was known then; a subject not
        // under follow-up at the cutoff has no forecast to make.
        let known = match args.forecast_cutoff {
            Some(cutoff) => {
                if cutoff <= subject.entry || cutoff > subject.exit {
                    skipped += 1;
                    continue;
                }
                subject
                    .prefix(cutoff, &predictor.mark_kinds)
                    .map_err(|e| e.to_string())?
            }
            None => subject.clone(),
        };
        let mut table = Array2::<f64>::zeros((records[i].len(), predictor.covariate_names.len()));
        for (r, record) in records[i].iter().enumerate() {
            for (j, value) in record.iter().enumerate() {
                table[[r, j]] = *value;
            }
        }
        let horizons: Vec<f64> = args
            .horizons_after_exit
            .iter()
            .map(|h| known.exit + h)
            .collect();
        let f = forecast_history(
            &predictor,
            &HistoryForecastRequest {
                history: &known,
                covariates: table.view(),
                stratum: strata[i],
                horizons: &horizons,
                future: &[],
            },
        )
        .map_err(|e| e.to_string())?;
        let mut entry = forecast_json(&f);
        entry["id"] = json!(known.id);
        forecasts.push(entry);
    }
    let mut document = Map::new();
    document.insert("marks".to_string(), json!(predictor.mark_names));
    document.insert(
        "mark_kinds".to_string(),
        json!(predictor.mark_kinds.iter().map(|k| k.name()).collect::<Vec<_>>()),
    );
    document.insert("forecasts".to_string(), Value::Array(forecasts));
    if let Some(cutoff) = args.forecast_cutoff {
        document.insert("forecast_cutoff".to_string(), json!(cutoff));
        document.insert("forecast_skipped".to_string(), json!(skipped));
    }
    write_json_output(args.out.as_deref(), &Value::Object(document))
}

pub(crate) fn run_forecast_population(args: ForecastPopulationArgs) -> Result<(), String> {
    let predictor = EventHistoryPredictor::load(&args.model).map_err(|e| e.to_string())?;
    let (headers, rows) = read_csv(&args.covariates)?;
    let starts = column(&headers, &rows, "start", &args.covariates)?;
    let cells = predictor
        .covariate_names
        .iter()
        .map(|name| column(&headers, &rows, name, &args.covariates))
        .collect::<Result<Vec<_>, String>>()?;
    let future = starts
        .iter()
        .enumerate()
        .map(|(row, start)| {
            Ok(FutureSegment {
                start: parse_f64(start, "segment start")?,
                covariates: coded_row(&predictor, &cells, row)?,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    let f = population_forecast(
        &predictor,
        &PopulationForecastRequest {
            start: args.start,
            stratum: args.reference_stratum,
            horizons: &args.horizons,
            future: &future,
        },
    )
    .map_err(|e| e.to_string())?;
    let mut document = Map::new();
    document.insert("marks".to_string(), json!(predictor.mark_names));
    document.insert(
        "mark_kinds".to_string(),
        json!(predictor.mark_kinds.iter().map(|k| k.name()).collect::<Vec<_>>()),
    );
    document.insert("forecast".to_string(), forecast_json(&f));
    write_json_output(args.out.as_deref(), &Value::Object(document))
}

/// One covariate row coded against the saved schema: a continuous covariate
/// parses a number, and a categorical value is coded against the saved levels
/// rather than re-encoded from these records.
fn coded_row(
    predictor: &EventHistoryPredictor,
    cells: &[Vec<&str>],
    row: usize,
) -> Result<Vec<f64>, String> {
    predictor
        .covariate_names
        .iter()
        .zip(&predictor.covariate_levels)
        .zip(cells)
        .map(|((name, levels), column)| {
            let value = if levels.is_empty() {
                CovariateValue::Number(parse_f64(column[row], name)?)
            } else {
                CovariateValue::Label(column[row].to_string())
            };
            code_covariate_value(name, levels, value).map_err(|e| e.to_string())
        })
        .collect()
}
