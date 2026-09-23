//! #2961 A11 across surfaces: a model fitted and saved by `gam joint-events fit`
//! in its own process reloads in-process and forecasts bit for bit what
//! `gam joint-events forecast` writes, and what the in-memory fit forecasts.
//! Event rows may come in any order, and the CLI refuses exactly what the
//! model's table encoder refuses.

use gam::event_history::MarkKind;
use gam::event_history::joint::{
    EventTable, JointEventModel, JointForecast, JointTables, SubjectTable, fit_joint_event_model,
};
use std::process::{Command, Output};

fn gam(args: &[&str]) -> Output {
    Command::new(gam_test_support::gam_binary!())
        .args(args)
        .output()
        .expect("spawn gam CLI")
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

fn tables(subjects: &[(&str, f64, f64)], events: &[(&str, f64, &str)]) -> JointTables {
    JointTables {
        subjects: SubjectTable {
            id: subjects.iter().map(|s| s.0.to_string()).collect(),
            entry: subjects.iter().map(|s| s.1).collect(),
            exit: subjects.iter().map(|s| s.2).collect(),
        },
        events: EventTable {
            id: events.iter().map(|e| e.0.to_string()).collect(),
            time: events.iter().map(|e| e.1).collect(),
            mark: events.iter().map(|e| e.2.to_string()).collect(),
        },
        ..JointTables::default()
    }
}

/// The forecast of the one history in `history`.
fn forecast_one(model: &JointEventModel, history: &JointTables, horizons: &[f64]) -> JointForecast {
    let conditioned = model.condition(history).expect("condition");
    assert_eq!(conditioned.len(), 1);
    conditioned[0].forecast(horizons).expect("forecast")
}

fn bits(values: impl IntoIterator<Item = f64>) -> Vec<u64> {
    values.into_iter().map(f64::to_bits).collect()
}

/// Every number in a forecast field, flattened row-major, as bits.
fn json_bits(value: &serde_json::Value) -> Vec<u64> {
    match value {
        serde_json::Value::Array(items) => items.iter().flat_map(json_bits).collect(),
        serde_json::Value::Number(number) => {
            vec![number.as_f64().expect("a finite number").to_bits()]
        }
        other => vec![f64::NAN.to_bits(), other.to_string().len() as u64],
    }
}

#[test]
fn a_cli_saved_joint_model_forecasts_bit_identically_in_every_process() {
    let scratch = tempfile::tempdir().expect("scratch directory");
    let path = |name: &str| {
        scratch
            .path()
            .join(name)
            .to_str()
            .expect("UTF-8 path")
            .to_string()
    };
    std::fs::write(path("subjects.csv"), "id,entry,exit\na,0,4\nb,0,6\n").expect("write subjects");
    std::fs::write(
        path("events.csv"),
        "id,time,mark\na,4,cvd_death\nb,0,diagnosis\nb,1,visit\nb,5,visit\n",
    )
    .expect("write events");
    std::fs::write(path("new_subjects.csv"), "id,entry,exit\nnew,0.5,2.5\n")
        .expect("write new subjects");
    std::fs::write(path("new_events.csv"), "id,time,mark\nnew,1,visit\n")
        .expect("write new events");
    let marks = "diagnosis:once,cvd_death:terminal,other_death:terminal,visit:recurrent";

    let fit = gam(&[
        "joint-events",
        "fit",
        "--subjects",
        &path("subjects.csv"),
        "--events",
        &path("events.csv"),
        "--marks",
        marks,
        "--out",
        &path("model.json"),
    ]);
    assert_eq!(fit.status.code(), Some(0), "{}", stderr(&fit));
    let forecast = gam(&[
        "joint-events",
        "forecast",
        "--model",
        &path("model.json"),
        "--subjects",
        &path("new_subjects.csv"),
        "--events",
        &path("new_events.csv"),
        "--horizons",
        "0.25,3,40",
        "--out",
        &path("forecast.json"),
    ]);
    assert_eq!(forecast.status.code(), Some(0), "{}", stderr(&forecast));
    let written: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path("forecast.json")).expect("read forecast"))
            .expect("parse forecast");
    let cli = &written["forecasts"][0];
    assert_eq!(cli["id"], "new");

    let new = tables(&[("new", 0.5, 2.5)], &[("new", 1.0, "visit")]);
    let horizons = [0.25, 3.0, 40.0];
    let model = fit_joint_event_model(
        Some(vec![
            ("diagnosis".to_string(), MarkKind::Once),
            ("cvd_death".to_string(), MarkKind::Terminal),
            ("other_death".to_string(), MarkKind::Terminal),
            ("visit".to_string(), MarkKind::Recurrent),
        ]),
        &tables(
            &[("a", 0.0, 4.0), ("b", 0.0, 6.0)],
            &[
                ("a", 4.0, "cvd_death"),
                ("b", 0.0, "diagnosis"),
                ("b", 1.0, "visit"),
                ("b", 5.0, "visit"),
            ],
        ),
    )
    .expect("in-memory fit");
    let in_memory = forecast_one(&model, &new, &horizons);
    let reloaded = forecast_one(
        &JointEventModel::load(scratch.path().join("model.json").as_path())
            .expect("reload the CLI's model"),
        &new,
        &horizons,
    );
    // The once-only diagnosis against both terminal causes takes the bracketed
    // quadrature route, so the comparison covers its reported error too.
    assert!(reloaded.incidence_error.iter().any(|&e| e > 0.0));
    for forecast in [&in_memory, &reloaded] {
        assert_eq!(json_bits(&cli["horizons"]), bits(forecast.horizons.iter().copied()));
        assert_eq!(json_bits(&cli["survival"]), bits(forecast.survival.iter().copied()));
        assert_eq!(json_bits(&cli["incidence"]), bits(forecast.incidence.iter().copied()));
        assert_eq!(
            json_bits(&cli["incidence_error"]),
            bits(forecast.incidence_error.iter().copied())
        );
    }
}

/// The CLI keeps no table checks of its own. Event rows may come in any order,
/// and an invalid table is refused by the encoder, naming the record and
/// leaving no saved model behind.
#[test]
fn a_cli_fit_takes_event_rows_in_any_order_and_refuses_invalid_tables() {
    let scratch = tempfile::tempdir().expect("scratch directory");
    let write = |name: &str, text: &str| {
        let path = scratch.path().join(name);
        std::fs::write(&path, text).expect("write table");
        path.to_str().expect("UTF-8 path").to_string()
    };
    let path = |name: &str| {
        scratch
            .path()
            .join(name)
            .to_str()
            .expect("UTF-8 path")
            .to_string()
    };
    let fit = |subjects: &str, events: &str, out: &str| {
        gam(&[
            "joint-events",
            "fit",
            "--subjects",
            subjects,
            "--events",
            events,
            "--marks",
            "diagnosis:once,cvd_death:terminal,visit:recurrent",
            "--out",
            out,
        ])
    };
    let subjects = write("subjects.csv", "id,entry,exit\na,0,4\nb,0,6\n");
    let sorted = write(
        "sorted.csv",
        "id,time,mark\na,4,diagnosis\na,4,cvd_death\nb,1,visit\nb,5,visit\n",
    );
    // The same rows interleaved across subjects, the later visit first, and the
    // terminal event listed before its simultaneous diagnosis.
    let shuffled = write(
        "shuffled.csv",
        "id,time,mark\nb,5,visit\na,4,cvd_death\nb,1,visit\na,4,diagnosis\n",
    );
    for (events, model) in [(&sorted, path("sorted.json")), (&shuffled, path("shuffled.json"))] {
        let fitted = fit(&subjects, events, &model);
        assert_eq!(fitted.status.code(), Some(0), "{}", stderr(&fitted));
    }
    let saved = |name: &str| std::fs::read(scratch.path().join(name)).expect("read saved model");
    assert_eq!(saved("shuffled.json"), saved("sorted.json"));

    let new_subjects = write("new_subjects.csv", "id,entry,exit\nnew,0.5,2.5\n");
    let forecast = |events: &str| {
        let output = gam(&[
            "joint-events",
            "forecast",
            "--model",
            &path("sorted.json"),
            "--subjects",
            &new_subjects,
            "--events",
            events,
            "--horizons",
            "0.25,3",
        ]);
        assert_eq!(output.status.code(), Some(0), "{}", stderr(&output));
        output.stdout
    };
    assert_eq!(
        forecast(&write("new_reversed.csv", "id,time,mark\nnew,2,visit\nnew,1,visit\n")),
        forecast(&write("new_sorted.csv", "id,time,mark\nnew,1,visit\nnew,2,visit\n"))
    );

    let out = path("refused.json");
    let visit = write("visit.csv", "id,time,mark\na,1,visit\n");
    let duplicated = write("duplicated.csv", "id,entry,exit\na,0,4\na,0,6\n");
    let unknown = write("unknown.csv", "id,time,mark\nc,1,visit\n");
    // Positive control: the same visit fits against distinct, known subjects.
    let control = fit(&subjects, &visit, &path("control.json"));
    assert_eq!(control.status.code(), Some(0), "{}", stderr(&control));
    for (subjects, events, reason) in [
        (&subjects, &unknown, "subject \"c\" is not in the subjects table"),
        (&duplicated, &visit, "subject \"a\" has two rows"),
    ] {
        let refused = fit(subjects, events, &out);
        assert!(!refused.status.success(), "{reason}");
        assert!(stderr(&refused).contains(reason), "{reason}: {}", stderr(&refused));
        assert!(!std::path::Path::new(&out).exists(), "{reason}");
    }
}

/// A model file the reader refuses exits with the data code, the category
/// `gamfit` raises (`DataError`) for the same file; a model path that cannot be
/// read is the invocation's, so it exits with the formula code.
#[test]
fn a_cli_forecast_refuses_a_bad_model_file_with_its_category() {
    let scratch = tempfile::tempdir().expect("scratch directory");
    let write = |name: &str, text: &str| {
        let path = scratch.path().join(name);
        std::fs::write(&path, text).expect("write fixture");
        path.to_str().expect("UTF-8 path").to_string()
    };
    let subjects = write("subjects.csv", "id,entry,exit\na,0,4\nb,0,6\n");
    let events = write("events.csv", "id,time,mark\na,4,cvd_death\nb,1,visit\n");
    let model = scratch
        .path()
        .join("model.json")
        .to_str()
        .expect("UTF-8 path")
        .to_string();
    let fit = gam(&[
        "joint-events",
        "fit",
        "--subjects",
        &subjects,
        "--events",
        &events,
        "--marks",
        "cvd_death:terminal,visit:recurrent",
        "--out",
        &model,
    ]);
    assert_eq!(fit.status.code(), Some(0), "{}", stderr(&fit));
    // Only a subject still at risk has a forecast: `a`'s history ended in a
    // terminal event, which the forecast refuses by name.
    let at_risk_subjects = write("at_risk_subjects.csv", "id,entry,exit\nb,0,6\n");
    let at_risk_events = write("at_risk_events.csv", "id,time,mark\nb,1,visit\n");
    let forecast = |model: &str| {
        gam(&[
            "joint-events",
            "forecast",
            "--model",
            model,
            "--subjects",
            &at_risk_subjects,
            "--events",
            &at_risk_events,
            "--horizons",
            "1",
        ])
    };
    // Positive control: the saved model forecasts.
    let saved = forecast(&model);
    assert_eq!(saved.status.code(), Some(0), "{}", stderr(&saved));

    let data = Some(gam::ErrorCategory::Data.exit_code());
    for (text, reason) in [
        ("not a saved model", "is not a model document"),
        (
            r#"{"kind": "gam", "version": 1, "model": {}}"#,
            r#"of kind Some("gam")"#,
        ),
    ] {
        let refused = forecast(&write("refused.json", text));
        let message = stderr(&refused);
        assert_eq!(refused.status.code(), data, "{reason}: {message}");
        assert!(message.contains(reason), "{reason}: {message}");
    }
    let missing = scratch.path().join("missing.json");
    let unreadable = forecast(missing.to_str().expect("UTF-8 path"));
    assert_eq!(
        unreadable.status.code(),
        Some(gam::ErrorCategory::Formula.exit_code()),
        "{}",
        stderr(&unreadable)
    );
    assert!(
        stderr(&unreadable).contains(&missing.display().to_string()),
        "{}",
        stderr(&unreadable)
    );
}
