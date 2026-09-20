//! gam#3016 across surfaces: `gam latent-residual` writes, for a frame with no
//! response column, the conditional latent residual `ζ = (z − m(a))/√v(a)`
//! the saved model computes in process, and refuses a model whose fit consumed
//! no conditional latent law.

use gam::inference::model::FittedModel;
use std::path::Path;
use std::process::{Command, Output};

/// `Corr(z, x)`: the conditional-mean slope the calibration removes.
const M_SHIFT: f64 = 0.6;
/// The CSV writer's fixed twelve decimals round each value by at most half a
/// unit in the twelfth place.
const CSV_ROUNDING: f64 = 0.5e-12;

fn request_document(latent_measure: &str) -> String {
    format!(
        r#"{{
  "schema": "gam.fit-request",
  "schema_version": 1,
  "formula": "y ~ x",
  "config": {{"family": "bernoulli-marginal-slope", "slope_formula": "1", "z_column": "z",
             "latent_measure": "{latent_measure}"}}
}}"#
    )
}

/// `x, ζ ~ N(0,1)` from a fixed linear congruential stream,
/// `z_raw = 3 + 2·(m·x + √(1−m²)·ζ)` and `y = 1{-0.2 + 0.5·x + 0.6·ζ + ε > 0}`.
/// The held-out frame carries an `id` column and no `y`.
fn write_fixture(path: &Path, rows: usize, seed: u64, with_response: bool) {
    let mut state = seed;
    let mut uniform = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((state >> 11) as f64 + 0.5) / (1_u64 << 53) as f64
    };
    let mut normal = move || {
        let (u1, u2) = (uniform(), uniform());
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    };
    let residual_sd = (1.0 - M_SHIFT * M_SHIFT).sqrt();
    let mut writer = csv::Writer::from_path(path).expect("create fixture");
    if with_response {
        writer.write_record(["x", "z", "y"]).expect("write header");
    } else {
        writer.write_record(["id", "x", "z"]).expect("write header");
    }
    for i in 0..rows {
        let x = normal();
        let zeta = normal();
        let z = 3.0 + 2.0 * (M_SHIFT * x + residual_sd * zeta);
        let y = -0.2 + 0.5 * x + 0.6 * zeta + normal() > 0.0;
        let record = if with_response {
            [
                format!("{x:.17e}"),
                format!("{z:.17e}"),
                u8::from(y).to_string(),
            ]
        } else {
            [format!("row{i}"), format!("{x:.17e}"), format!("{z:.17e}")]
        };
        writer.write_record(record).expect("write fixture row");
    }
    writer.flush().expect("flush fixture");
}

fn gam(scratch: &Path, args: &[&Path]) -> Output {
    Command::new(gam_test_support::gam_binary!())
        .current_dir(scratch)
        .args(args)
        .output()
        .expect("spawn gam")
}

fn describe(output: &Output) -> String {
    format!(
        "status {}\n--- stdout ---\n{}\n--- stderr ---\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    )
}

fn fit(scratch: &Path, train: &Path, latent_measure: &str) -> std::path::PathBuf {
    let request = scratch.join(format!("{latent_measure}.json"));
    let model = scratch.join(format!("{latent_measure}.gam"));
    std::fs::write(&request, request_document(latent_measure)).expect("write request document");
    let output = gam(
        scratch,
        &[
            Path::new("fit"),
            train,
            Path::new("--request"),
            &request,
            Path::new("--out"),
            &model,
        ],
    );
    assert!(
        output.status.success(),
        "gam fit ({latent_measure}) failed: {}",
        describe(&output)
    );
    model
}

#[test]
fn latent_residual_cli_matches_the_saved_model_3016() {
    let scratch = tempfile::tempdir().expect("scratch dir");
    let dir = scratch.path();
    let train = dir.join("train.csv");
    let held_out = dir.join("held_out.csv");
    let out = dir.join("residual.csv");
    write_fixture(&train, 800, 3016, true);
    write_fixture(&held_out, 300, 6103, false);

    let model_path = fit(dir, &train, "conditional-location-scale");
    let output = gam(
        dir,
        &[
            Path::new("latent-residual"),
            &model_path,
            &held_out,
            Path::new("--out"),
            &out,
            Path::new("--id-column"),
            Path::new("id"),
        ],
    );
    assert!(
        output.status.success(),
        "gam latent-residual failed: {}",
        describe(&output)
    );

    let mut reader = csv::Reader::from_path(&out).expect("open residual csv");
    let headers = reader.headers().expect("residual csv header").clone();
    assert_eq!(
        headers.iter().collect::<Vec<_>>(),
        ["id", "residual"],
        "the residual csv carries the id column and one residual column"
    );
    let mut ids = Vec::new();
    let mut cli = Vec::new();
    for record in reader.records() {
        let record = record.expect("residual csv row");
        ids.push(record[0].to_string());
        cli.push(record[1].parse::<f64>().expect("residual is a number"));
    }
    assert_eq!(ids, (0..300).map(|i| format!("row{i}")).collect::<Vec<_>>());

    let model = FittedModel::load_from_path(&model_path).expect("load the CLI-saved model");
    let dataset = gam_data::load_csvwith_inferred_schema(&held_out).expect("load held-out frame");
    let in_process = model
        .latent_conditional_residual(dataset.values.view(), &dataset.column_map())
        .expect("in-process residual")
        .expect("a conditional-law fit returns a residual");
    assert_eq!(cli.len(), in_process.len(), "one residual per held-out row");
    let worst = cli
        .iter()
        .zip(in_process.iter())
        .map(|(written, computed)| (written - computed).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        worst <= CSV_ROUNDING * (1.0 + 1.0e-6),
        "the CLI residual differs from the saved model's by {worst:e}, more than the csv rounding"
    );

    let global = fit(dir, &train, "global-empirical");
    let refused = gam(
        dir,
        &[
            Path::new("latent-residual"),
            &global,
            &held_out,
            Path::new("--out"),
            &out,
        ],
    );
    assert!(
        !refused.status.success(),
        "a global-empirical model has no residual: {}",
        describe(&refused)
    );
    let stderr = String::from_utf8_lossy(&refused.stderr);
    assert!(
        stderr.contains("conditional-location-scale"),
        "the refusal names the latent law the command needs: {stderr}"
    );
}
