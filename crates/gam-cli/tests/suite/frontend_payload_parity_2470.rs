//! #2470 — one canonical `gam.fit-request` document must save one model, no
//! matter which front end submitted it.
//!
//! The two front ends already share the payload *assemblers*
//! (`assemble_standard_payload` and friends), but they did not share the copy
//! of the request metadata that no assembler can derive from a fitted result:
//! `group_metadata`, `training_table_kind`, `inference_notes` and
//! `informational_notes`. The Python binding reaches that copy through
//! `fit_formula_to_payload`; every CLI save
//! route open-coded it, and every one of them omitted `training_table_kind`, so
//! the same request document persisted `"polars"` from Python and the
//! `"unknown"` default from `gam fit`.
//!
//! This test submits ONE document to BOTH routes and compares the saved
//! metadata field by field. It is deliberately not a size or shape check: it
//! reads the values, and it also pins the CLI value to the document's own
//! `"polars"` so a regression that reinstates the default cannot pass by making
//! both sides agree on being wrong.

use gam::config_resolve::parse_fit_request_json;
use gam::inference::model::FittedModel;
use gam::inference::model_payload_builders::fit_formula_to_payload;
use std::path::Path;
use std::process::Command;

/// The single canonical request both front ends are handed. `training_table_kind`
/// is transport-neutral by construction: the document carries it precisely so a
/// frontend that knows its caller's container can record it, and every other
/// frontend replays the same saved model from the same document.
const REQUEST_JSON: &str = r#"{
  "schema": "gam.fit-request",
  "schema_version": 1,
  "formula": "y ~ x",
  "config": {
    "family": "gaussian",
    "training_table_kind": "polars",
    "group_metadata": {"cohort": "frontend-parity-2470"}
  }
}"#;

fn write_training_fixture(path: &Path) {
    let mut writer = csv::Writer::from_path(path).expect("create training fixture");
    writer
        .write_record(["x", "y"])
        .expect("write training header");
    for i in 0..64 {
        let x = i as f64 / 63.0;
        let deterministic_noise = 0.02 * ((i % 5) as f64 - 2.0);
        let y = 1.5 + 2.25 * x + deterministic_noise;
        writer
            .write_record([x.to_string(), y.to_string()])
            .expect("write training row");
    }
    writer.flush().expect("flush training fixture");
}

#[test]
fn frontend_request_metadata_parity_2470() {
    let scratch = tempfile::tempdir().expect("create isolated scratch directory");
    let train_path = scratch.path().join("train.csv");
    let request_path = scratch.path().join("request.json");
    let model_path = scratch.path().join("cli_model.gam");
    write_training_fixture(&train_path);
    std::fs::write(&request_path, REQUEST_JSON).expect("write fit request document");

    // Route A: the CLI, as a fresh process, saving through `gam fit --out`.
    let output = Command::new(gam_test_support::gam_binary!())
        .current_dir(scratch.path())
        .arg("fit")
        .arg(&train_path)
        .arg("--request")
        .arg(&request_path)
        .arg("--out")
        .arg(&model_path)
        .output()
        .expect("spawn gam fit");
    assert!(
        output.status.success(),
        "gam fit failed with status {}\n--- stdout ---\n{}\n--- stderr ---\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
    let cli_model = FittedModel::load_from_path(&model_path).expect("load CLI-saved model");
    let cli = cli_model.payload();

    // Route B: the shared service the Python binding's `fit_dataset` calls, in
    // process, on the same document and the same table.
    let resolved = parse_fit_request_json(REQUEST_JSON).expect("resolve fit request document");
    let dataset =
        gam_data::load_csvwith_inferred_schema(&train_path).expect("load training fixture");
    let shared = fit_formula_to_payload(resolved.formula, &dataset, &resolved.fit_config)
        .expect("shared formula-to-payload service");

    // The document said "polars". A CLI save that silently substitutes the
    // "unknown" default is the #2470 defect, and it is invisible to any check
    // that only compares the two routes to each other.
    assert_eq!(
        cli.training_table_kind, "polars",
        "the CLI dropped the request document's training_table_kind"
    );
    assert_eq!(
        cli.training_table_kind, shared.training_table_kind,
        "training_table_kind diverged between the CLI and the shared save service"
    );
    assert_eq!(
        cli.group_metadata, shared.group_metadata,
        "group_metadata diverged between the CLI and the shared save service"
    );
    assert_eq!(
        cli.inference_notes, shared.inference_notes,
        "inference_notes diverged between the CLI and the shared save service"
    );
    assert_eq!(
        cli.informational_notes, shared.informational_notes,
        "informational_notes diverged between the CLI and the shared save service"
    );
}
