//! `gam fit-events`: fit an event-history model from three CSV tables and
//! write a JSON summary with optional forecasts.

use crate::cli_args::FitEventsArgs;
use gam::event_history::{
    CovariateCells, CovariateSegment, Event, EventHistoryCohort, ForecastRequest, FutureSegment,
    MarkKind, PopulationForecastRequest, ReferenceStrata, SubjectHistory,
    fit_event_history_formulas, forecast, latent_state, pit_uniform_distance, population_forecast,
    predictive_pit, resolve_mark_vocabulary,
};
use gam::families::custom_family::BlockwiseFitOptions;
use ndarray::Array2;
use serde_json::{Map, Value, json};
use std::collections::HashMap;
use std::path::Path;

fn read_csv(path: &Path) -> Result<(Vec<String>, Vec<Vec<String>>), String> {
    let mut reader = csv::Reader::from_path(path)
        .map_err(|error| format!("cannot open {}: {error}", path.display()))?;
    let headers: Vec<String> = reader
        .headers()
        .map_err(|error| format!("{}: {error}", path.display()))?
        .iter()
        .map(|h| h.trim().to_string())
        .collect();
    let mut rows = Vec::new();
    for record in reader.records() {
        let record = record.map_err(|error| format!("{}: {error}", path.display()))?;
        rows.push(record.iter().map(|v| v.trim().to_string()).collect());
    }
    Ok((headers, rows))
}

fn column<'a>(
    headers: &[String],
    rows: &'a [Vec<String>],
    name: &str,
    path: &Path,
) -> Result<Vec<&'a str>, String> {
    let index = headers
        .iter()
        .position(|h| h == name)
        .ok_or_else(|| format!("{} has no column {name:?}", path.display()))?;
    rows.iter()
        .map(|row| {
            row.get(index)
                .map(|v| v.as_str())
                .ok_or_else(|| format!("{}: a row is shorter than its header", path.display()))
        })
        .collect()
}

fn parse_f64(value: &str, what: &str) -> Result<f64, String> {
    value
        .parse::<f64>()
        .map_err(|_| format!("{what}: {value:?} is not a number"))
}

fn forecast_json(f: &gam::event_history::Forecast) -> Value {
    json!({
        "horizons": f.horizons,
        "survival": f.survival,
        "expected_counts": f.expected_counts.rows().into_iter().map(|r| r.to_vec()).collect::<Vec<_>>(),
    })
}

pub(crate) fn run_fit_events(args: FitEventsArgs) -> Result<(), String> {
    let (subject_headers, subject_rows) = read_csv(&args.subjects)?;
    let ids = column(&subject_headers, &subject_rows, "id", &args.subjects)?;
    let entries = column(&subject_headers, &subject_rows, "entry", &args.subjects)?;
    let exits = column(&subject_headers, &subject_rows, "exit", &args.subjects)?;
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
    // The mark vocabulary: declared with kinds, or the observed marks, all
    // recurrent; the cohort resolver also indexes every event's mark.
    if args.marks.is_empty() && event_marks.is_empty() {
        return Err(
            "the events table has no rows, so the marks must be declared with --marks name:kind,..."
                .to_string(),
        );
    }
    let declared = if args.marks.is_empty() {
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
    let (mark_names, mark_kinds, event_mark_indices) =
        resolve_mark_vocabulary(declared, &event_marks).map_err(|e| e.to_string())?;
    for ((id, time), mark_index) in event_ids
        .iter()
        .zip(event_times.iter())
        .zip(event_mark_indices.iter())
    {
        let subject = *index
            .get(*id)
            .ok_or_else(|| format!("event subject {id:?} is not in the subjects table"))?;
        subjects[subject].events.push(Event {
            time: parse_f64(time, "event time")?,
            mark: *mark_index,
        });
    }
    let (cov_headers, cov_rows) = read_csv(&args.covariates)?;
    let cov_ids = column(&cov_headers, &cov_rows, "id", &args.covariates)?;
    let cov_starts = column(&cov_headers, &cov_rows, "start", &args.covariates)?;
    let covariate_names: Vec<String> = cov_headers
        .iter()
        .filter(|h| h.as_str() != "id" && h.as_str() != "start")
        .cloned()
        .collect();
    let mut table = Array2::<f64>::zeros((cov_rows.len(), covariate_names.len()));
    let mut covariate_levels = Vec::with_capacity(covariate_names.len());
    for (j, name) in covariate_names.iter().enumerate() {
        let values = column(&cov_headers, &cov_rows, name, &args.covariates)?;
        let (codes, levels) = CovariateCells::from_text(&values).encode();
        for (i, code) in codes.iter().enumerate() {
            table[[i, j]] = *code;
        }
        covariate_levels.push(levels);
    }
    for (row, (id, start)) in cov_ids.iter().zip(cov_starts.iter()).enumerate() {
        let subject = *index
            .get(*id)
            .ok_or_else(|| format!("covariate subject {id:?} is not in the subjects table"))?;
        subjects[subject].segments.push(CovariateSegment {
            start: parse_f64(start, "segment start")?,
            row,
        });
    }
    let mut cohort = EventHistoryCohort {
        mark_names: mark_names.clone(),
        mark_kinds: mark_kinds.clone(),
        covariate_names: covariate_names.clone(),
        covariate_levels: covariate_levels.clone(),
        covariates: table,
        subjects,
    };
    // One formula for every mark, or one per mark by name.
    let formulas: Vec<String> = match (&args.formula, args.mark_formula.is_empty()) {
        (Some(formula), true) => vec![formula.clone()],
        (None, false) => {
            let mut per_mark: Vec<Option<String>> = vec![None; mark_names.len()];
            for entry in &args.mark_formula {
                let (name, rhs) = entry
                    .split_once('=')
                    .ok_or_else(|| format!("--mark-formula entry {entry:?} is not NAME=RHS"))?;
                let index = mark_names
                    .iter()
                    .position(|m| m == name.trim())
                    .ok_or_else(|| {
                        format!(
                            "--mark-formula names mark {:?}, not in the vocabulary {mark_names:?}",
                            name.trim()
                        )
                    })?;
                if per_mark[index].replace(rhs.trim().to_string()).is_some() {
                    return Err(format!(
                        "--mark-formula gives mark {:?} two formulas",
                        name.trim()
                    ));
                }
            }
            per_mark
                .into_iter()
                .zip(mark_names.iter())
                .map(|(formula, name)| {
                    formula.ok_or_else(|| format!("--mark-formula gives mark {name:?} no formula"))
                })
                .collect::<Result<Vec<_>, _>>()?
        }
        (Some(_), false) => {
            return Err(
                "give either --formula or one --mark-formula per mark, not both".to_string(),
            );
        }
        (None, true) => {
            return Err(
                "a log-intensity formula is needed: --formula, or one --mark-formula per mark"
                    .to_string(),
            );
        }
    };
    // The reference population, when the baselines are to be the incidence
    // among those still at risk rather than the rate over the cohort as it
    // started. The strata's profiles are rows of the covariate table; a
    // column of the subjects table assigns subjects to them.
    let mut subject_stratum: Vec<usize> = vec![0; cohort.subjects.len()];
    let reference = if args.reference_row.is_empty() {
        if args.reference_stratum.is_some() {
            return Err(
                "--reference-stratum needs at least one --reference-row to assign subjects to"
                    .to_string(),
            );
        }
        None
    } else {
        let assigned = match &args.reference_stratum {
            None => {
                if args.reference_row.len() != 1 {
                    return Err(format!(
                        "{} --reference-row profiles need --reference-stratum to say which subject is in which",
                        args.reference_row.len()
                    ));
                }
                vec![0usize; cohort.subjects.len()]
            }
            Some(name) => {
                let values = column(&subject_headers, &subject_rows, name, &args.subjects)?;
                values
                    .iter()
                    .map(|v| {
                        let index = v.parse::<usize>().map_err(|_| format!(
                            "reference stratum {v:?} in column {name:?} must be a nonnegative integer index"))?;
                        if index >= args.reference_row.len() {
                            return Err(format!("reference stratum {index} is outside the {} reference profiles", args.reference_row.len()));
                        }
                        Ok(index)
                    })
                    .collect::<Result<Vec<_>, String>>()?
            }
        };
        subject_stratum = assigned;
        Some(ReferenceStrata {
            rows: args.reference_row.clone(),
            subject: subject_stratum.clone(),
        })
    };
    let has_reference = reference.is_some();
    let fit = fit_event_history_formulas(
        &mut cohort,
        &formulas,
        BlockwiseFitOptions::default(),
        reference,
    )
    .map_err(|e| e.to_string())?;

    let mut summary = Map::new();
    summary.insert("marks".to_string(), json!(mark_names));
    summary.insert(
        "mark_kinds".to_string(),
        json!(mark_kinds.iter().map(|k| k.name()).collect::<Vec<_>>()),
    );
    summary.insert("covariates".to_string(), json!(covariate_names));
    summary.insert("covariate_levels".to_string(), json!(covariate_levels));
    summary.insert(
        "formula".to_string(),
        if formulas.len() == 1 {
            json!(formulas[0])
        } else {
            json!(formulas)
        },
    );
    summary.insert("rank".to_string(), json!(fit.rank()));
    // Whether the rank is the one the evidence selected, or the certified
    // incumbent a decision the quadrature could not resolve stopped at.
    summary.insert(
        "rank_verdict".to_string(),
        json!(if fit.unresolved_growth().is_some() {
            "growth_unresolved_at_lebesgue_bound"
        } else {
            "evidence"
        }),
    );
    if has_reference {
        // The reference grid's fixed-coefficient moves and its certificate,
        // in posterior standard deviations.
        summary.insert(
            "reference_refinements".to_string(),
            json!(fit.reference_refinements),
        );
        summary.insert(
            "reference_masks".to_string(),
            json!(fit.centring.as_ref().map_or(0, |c| c.masks)),
        );
        summary.insert(
            "reference_certificate".to_string(),
            json!(fit.reference_certificate),
        );
    }
    summary.insert("atom_evidence".to_string(), json!(fit.atom_evidence));
    summary.insert(
        "rank_path".to_string(),
        json!(
            fit.rank_path
                .iter()
                .map(|step| {
                    json!({
                        "rank": step.rank,
                        "score_eigenvalue": step.score_eigenvalue,
                        "standardised_gain": step.standardised_gain,
                        "proposed_rate": step.proposed_rate,
                        "at_resolution_limit": step.at_resolution_limit,
                        "rate_held": step.rate_held,
                        "ridge_log_lambda": step.ridge_log_lambda,
                        "evidence_gain": step.evidence_gain,
                        "log_likelihood_gain": step.log_likelihood_gain,
                        "accepted": step.accepted,
                        "converged": step.converged,
                        "growth_unresolved": step.growth_unresolved.as_ref().map(|growth| json!({
                            "gauss_hermite_order": growth.gauss_hermite_order,
                            "integral": growth.integral.name(),
                            "reason": growth.reason,
                        })),
                    })
                })
                .collect::<Vec<_>>()
        ),
    );
    let rows = |matrix: &ndarray::Array2<f64>| -> Vec<Vec<f64>> {
        matrix.rows().into_iter().map(|r| r.to_vec()).collect()
    };
    summary.insert("covariance".to_string(), json!(rows(&fit.covariance)));
    summary.insert("eigenvalues".to_string(), json!(fit.eigenvalues.to_vec()));
    summary.insert(
        "eigenvalue_sd".to_string(),
        json!(fit.eigenvalue_sd.to_vec()),
    );
    summary.insert("eigenvectors".to_string(), json!(rows(&fit.eigenvectors)));
    summary.insert("effective_rank".to_string(), json!(fit.effective_rank));
    if fit.rank() > 0 {
        let mut states = Vec::with_capacity(cohort.subjects.len());
        for (i, subject) in cohort.subjects.iter().enumerate() {
            let state = latent_state(&fit, &cohort, subject, subject_stratum[i])
                .map_err(|e| e.to_string())?;
            states.push(json!({
                "id": subject.id,
                "time": state.times,
                "mean": rows(&state.mean),
                "covariance": state.covariance.iter().map(rows).collect::<Vec<_>>(),
            }));
        }
        summary.insert("latent_states".to_string(), Value::Array(states));
    }
    summary.insert("log_likelihood".to_string(), json!(fit.fit.log_likelihood));
    summary.insert(
        "reml_score".to_string(),
        json!(
            fit.fit
                .comparable_reml_score()
                .map_err(|err| format!("failed to compute comparable REML score: {err}"))?
        ),
    );
    summary.insert("raw_reml_score".to_string(), json!(fit.fit.reml_score()));
    summary.insert(
        "outer_iterations".to_string(),
        json!(fit.fit.outer_iterations),
    );
    summary.insert("time_scale".to_string(), json!(fit.time_scale));
    summary.insert("loadings".to_string(), json!(rows(&fit.loadings)));
    summary.insert("rates".to_string(), json!(fit.rates));
    summary.insert("rate_held".to_string(), json!(fit.rate_held));
    summary.insert("atom_log_lambdas".to_string(), json!(fit.atom_log_lambdas));
    summary.insert(
        "coefficients".to_string(),
        json!(
            (0..fit.marks())
                .map(|d| fit.mark_coefficients(d).to_vec())
                .collect::<Vec<_>>()
        ),
    );
    let q = &fit.quadrature;
    summary.insert(
        "quadrature".to_string(),
        json!({
            "gauss_hermite_order": q.gauss_hermite_order,
            "mesh_refinement": q.mesh_refinement,
            "log_likelihood": q.log_likelihood,
            "gauss_hermite_check": {
                "order": q.gauss_hermite.candidate,
                "coefficient_shift": q.gauss_hermite.coefficient_shift,
                "log_likelihood": q.gauss_hermite.log_likelihood,
            },
            "mesh_check": {
                "refinement": q.mesh.candidate,
                "coefficient_shift": q.mesh.coefficient_shift,
                "log_likelihood": q.mesh.log_likelihood,
            },
        }),
    );
    let mut pits = Vec::new();
    for (i, subject) in cohort.subjects.iter().enumerate() {
        pits.extend(
            predictive_pit(&fit, &cohort, subject, subject_stratum[i])
                .map_err(|e| e.to_string())?,
        );
    }
    summary.insert("pit_spells".to_string(), json!(pits.len()));
    summary.insert(
        "pit_events".to_string(),
        json!(pits.iter().filter(|p| p.observed).count()),
    );
    summary.insert(
        "pit_distance".to_string(),
        json!(pit_uniform_distance(&pits)),
    );
    if !args.horizons_after_exit.is_empty() {
        let mut forecasts = Vec::with_capacity(cohort.subjects.len());
        let mut skipped = 0usize;
        for (i, subject) in cohort.subjects.iter().enumerate() {
            // With a cutoff, the history is what was known then; a subject
            // not under follow-up at the cutoff has no forecast to make.
            let known: SubjectHistory = match args.forecast_cutoff {
                Some(cutoff) => {
                    if cutoff <= subject.entry || cutoff > subject.exit {
                        skipped += 1;
                        continue;
                    }
                    subject
                        .prefix(cutoff, &mark_kinds)
                        .map_err(|e| e.to_string())?
                }
                None => subject.clone(),
            };
            let subject = &known;
            let horizons: Vec<f64> = args
                .horizons_after_exit
                .iter()
                .map(|h| subject.exit + h)
                .collect();
            let f = forecast(
                &fit,
                &cohort,
                &ForecastRequest {
                    history: subject,
                    horizons: &horizons,
                    future: &[],
                    stratum: subject_stratum[i],
                },
            )
            .map_err(|e| e.to_string())?;
            // The same window from the stationary prior at the subject's
            // covariates at exit: what the model says without its history.
            let row = subject.covariate_row_at(subject.exit, false);
            let alone = population_forecast(
                &fit,
                &cohort,
                &PopulationForecastRequest {
                    start: subject.exit,
                    horizons: &horizons,
                    future: &[FutureSegment {
                        start: subject.exit,
                        covariates: cohort.covariates.row(row).to_vec(),
                    }],
                    stratum: subject_stratum[i],
                },
            )
            .map_err(|e| e.to_string())?;
            let mut entry = forecast_json(&f);
            entry["id"] = json!(subject.id);
            entry["without_history"] = json!({
                "survival": alone.survival,
                "expected_counts": alone.expected_counts.rows().into_iter().map(|r| r.to_vec()).collect::<Vec<_>>(),
            });
            forecasts.push(entry);
        }
        summary.insert("forecasts".to_string(), Value::Array(forecasts));
        if let Some(cutoff) = args.forecast_cutoff {
            summary.insert("forecast_cutoff".to_string(), json!(cutoff));
            summary.insert("forecast_skipped".to_string(), json!(skipped));
        }
    }
    let text = serde_json::to_string_pretty(&Value::Object(summary))
        .map_err(|error| format!("serialising the summary: {error}"))?;
    match &args.out {
        Some(path) => std::fs::write(path, text)
            .map_err(|error| format!("writing {}: {error}", path.display())),
        None => {
            use std::io::Write;
            let mut stdout = std::io::stdout().lock();
            stdout
                .write_all(text.as_bytes())
                .and_then(|()| stdout.write_all(b"\n"))
                .map_err(|error| format!("writing the summary to stdout: {error}"))
        }
    }
}
