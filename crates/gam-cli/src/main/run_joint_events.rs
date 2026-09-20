//! `gam joint-events`: fit the joint event model to subjects and events tables
//! and save it, or condition a saved model on histories and forecast (#2961).
//! Both actions hand the tables to the one Rust model path the Python library
//! also calls; the vocabulary, identifiers and records are checked there.

use crate::cli_args::{
    JointEventsAction, JointEventsArgs, JointEventsFitArgs, JointEventsForecastArgs,
};
use gam::event_history::MarkKind;
use gam::event_history::joint::{
    EventTable, JointEventModel, JointTables, SubjectTable, fit_joint_event_model,
};
use ndarray::Array2;
use serde_json::json;
use std::path::Path;

/// The named columns of a CSV table, in the order named.
fn read_columns(path: &Path, names: &[&str]) -> Result<Vec<Vec<String>>, String> {
    let mut reader = csv::Reader::from_path(path)
        .map_err(|error| format!("cannot open {}: {error}", path.display()))?;
    let headers: Vec<String> = reader
        .headers()
        .map_err(|error| format!("{}: {error}", path.display()))?
        .iter()
        .map(|h| h.trim().to_string())
        .collect();
    let indices = names
        .iter()
        .map(|name| {
            headers
                .iter()
                .position(|h| h == name)
                .ok_or_else(|| format!("{} has no column {name:?}", path.display()))
        })
        .collect::<Result<Vec<usize>, String>>()?;
    let mut columns = vec![Vec::new(); names.len()];
    for record in reader.records() {
        let record = record.map_err(|error| format!("{}: {error}", path.display()))?;
        for (column, &index) in columns.iter_mut().zip(&indices) {
            let value = record
                .get(index)
                .ok_or_else(|| format!("{}: a row is shorter than its header", path.display()))?;
            column.push(value.trim().to_string());
        }
    }
    Ok(columns)
}

fn numbers(cells: &[String], what: &str) -> Result<Vec<f64>, String> {
    cells
        .iter()
        .map(|value| {
            value
                .parse::<f64>()
                .map_err(|_| format!("{what}: {value:?} is not a number"))
        })
        .collect()
}

/// The subjects table `(id, entry, exit)` and the events table `(id, time,
/// mark)` as the model's tables.
fn read_tables(subjects: &Path, events: &Path) -> Result<JointTables, String> {
    let subject_columns = read_columns(subjects, &["id", "entry", "exit"])?;
    let event_columns = read_columns(events, &["id", "time", "mark"])?;
    Ok(JointTables {
        subjects: SubjectTable {
            id: subject_columns[0].clone(),
            entry: numbers(&subject_columns[1], "entry")?,
            exit: numbers(&subject_columns[2], "exit")?,
        },
        events: EventTable {
            id: event_columns[0].clone(),
            time: numbers(&event_columns[1], "event time")?,
            mark: event_columns[2].clone(),
        },
        ..JointTables::default()
    })
}

fn fit(args: JointEventsFitArgs) -> Result<(), String> {
    let tables = read_tables(&args.subjects, &args.events)?;
    let marks = if args.marks.is_empty() {
        None
    } else {
        let mut pairs = Vec::with_capacity(args.marks.len());
        for spec in &args.marks {
            let (name, kind) = spec
                .split_once(':')
                .ok_or_else(|| format!("--marks entry {spec:?} is not name:kind"))?;
            pairs.push((
                name.trim().to_string(),
                MarkKind::parse(kind).map_err(|e| e.to_string())?,
            ));
        }
        Some(pairs)
    };
    let model = fit_joint_event_model(marks, &tables).map_err(|e| e.to_string())?;
    model.save(&args.out).map_err(|e| e.to_string())
}

fn rows(matrix: &Array2<f64>) -> Vec<Vec<f64>> {
    matrix.rows().into_iter().map(|r| r.to_vec()).collect()
}

fn forecast(args: JointEventsForecastArgs) -> Result<(), String> {
    let model = JointEventModel::load(&args.model).map_err(|e| e.to_string())?;
    let tables = read_tables(&args.subjects, &args.events)?;
    let conditioned = model.condition(&tables).map_err(|e| e.to_string())?;
    let mut forecasts = Vec::with_capacity(conditioned.len());
    for history in &conditioned {
        let f = history
            .forecast(&args.horizons)
            .map_err(|e| e.to_string())?;
        forecasts.push(json!({
            "id": history.id(),
            "time": history.exit(),
            "horizons": f.horizons,
            "survival": f.survival,
            "incidence": rows(&f.incidence),
            "incidence_error": rows(&f.incidence_error),
        }));
    }
    let summary = json!({
        "marks": model.mark_names(),
        "mark_kinds": model.mark_kinds().iter().map(|k| k.name()).collect::<Vec<_>>(),
        "forecasts": forecasts,
    });
    let text = serde_json::to_string_pretty(&summary)
        .map_err(|error| format!("serialising the forecasts: {error}"))?;
    match &args.out {
        Some(path) => std::fs::write(path, text)
            .map_err(|error| format!("writing {}: {error}", path.display())),
        None => {
            use std::io::Write;
            let mut stdout = std::io::stdout().lock();
            stdout
                .write_all(text.as_bytes())
                .and_then(|()| stdout.write_all(b"\n"))
                .map_err(|error| format!("writing the forecasts to stdout: {error}"))
        }
    }
}

pub(crate) fn run_joint_events(args: JointEventsArgs) -> Result<(), String> {
    match args.action {
        JointEventsAction::Fit(args) => fit(args),
        JointEventsAction::Forecast(args) => forecast(args),
    }
}
