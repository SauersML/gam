//! gam#2956: `FitConfig::resolve` refused `latent_measure` on a Bernoulli
//! marginal-slope request selected by `slope_formula` and `z_column` with no
//! explicit family ("latent_measure applies to marginal-slope fits only"),
//! although materialization fits exactly that request as marginal slope. One
//! request document, no family: it must resolve, fit, and save the latent law
//! it asked for, while a request that is not a marginal-slope fit is still
//! refused by name.

use gam::config_resolve::parse_fit_request_json;
use gam::families::bms::LatentMeasureKind;
use gam::inference::model_payload_builders::fit_formula_to_payload;
use std::path::Path;

fn request_document(config: &str) -> String {
    format!(
        r#"{{
  "schema": "gam.fit-request",
  "schema_version": 1,
  "formula": "y ~ x",
  "config": {config}
}}"#
    )
}

/// A deterministic probit fixture: `x` on a grid, `z` an unconditioned standard
/// normal score from a fixed linear congruential stream, and
/// `y = 1{-0.2 + 0.5·x + 0.8·z + ε > 0}`.
fn write_training_fixture(path: &Path) {
    let mut state: u64 = 2956;
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
    let mut writer = csv::Writer::from_path(path).expect("create training fixture");
    writer.write_record(["x", "z", "y"]).expect("write training header");
    let rows = 800;
    for i in 0..rows {
        let x = -2.0 + 4.0 * i as f64 / (rows - 1) as f64;
        let z = normal();
        let y = -0.2 + 0.5 * x + 0.8 * z + normal() > 0.0;
        writer
            .write_record([format!("{x:.17e}"), format!("{z:.17e}"), u8::from(y).to_string()])
            .expect("write training row");
    }
    writer.flush().expect("flush training fixture");
}

#[test]
fn a_request_document_without_a_family_accepts_latent_measure_2956() {
    let scratch = tempfile::tempdir().expect("scratch dir");
    let train_path = scratch.path().join("train.csv");
    write_training_fixture(&train_path);
    let dataset = gam_data::load_csvwith_inferred_schema(&train_path).expect("load training fixture");

    for (requested, expect_global_empirical) in [("global-empirical", true), ("standard-normal", false)] {
        let document = request_document(&format!(
            r#"{{"slope_formula": "1", "z_column": "z", "latent_measure": "{requested}"}}"#
        ));
        let resolved = parse_fit_request_json(&document).unwrap_or_else(|error| {
            panic!("a z_column-selected marginal-slope request must accept latent_measure = {requested}: {error}")
        });
        let payload = fit_formula_to_payload(resolved.formula, &dataset, &resolved.fit_config)
            .unwrap_or_else(|error| panic!("latent_measure = {requested} fits: {error}"));
        let saved = payload.latent_measure.as_ref();
        assert_eq!(
            matches!(saved, Some(LatentMeasureKind::GlobalEmpirical { .. })),
            expect_global_empirical,
            "the saved model must record the requested latent law {requested}"
        );
        assert_eq!(
            matches!(saved, Some(LatentMeasureKind::StandardNormal)),
            !expect_global_empirical,
            "the saved model must record the requested latent law {requested}"
        );
    }

    let not_marginal_slope =
        request_document(r#"{"family": "gaussian", "latent_measure": "global-empirical"}"#);
    let refusal = parse_fit_request_json(&not_marginal_slope)
        .and_then(|request| request.fit_config.resolve().map(|_| ()))
        .expect_err("latent_measure on a Gaussian request must be refused");
    assert!(
        refusal.contains("latent_measure applies to marginal-slope fits only"),
        "the refusal must name the control: {refusal}"
    );
}
