//! `gam fit --noise-offset-column` on a Bernoulli marginal-slope fit. Under the
//! marginal-slope families the noise offset is the slope predictor's offset: the
//! library materializer reads it there, the saved payload records it and
//! `gam predict` replays it. The CLI must not refuse it for want of
//! `--predict-noise`, while a fit on the standard route still meets the
//! library's refusal of a column it would drop (the CLI keeps no mirror of the
//! route predicate).

use std::path::Path;
use std::process::{Command, Output};

/// `x, ζ ~ N(0,1)` from a fixed linear congruential stream, `z = 0.5·x + ζ`,
/// a slope offset `off = 0.25·x` and `y = 1{-0.2 + 0.5·x + 0.6·z + ε > 0}`.
fn write_fixture(path: &Path, rows: usize, seed: u64) {
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
    let mut writer = csv::Writer::from_path(path).expect("create fixture");
    writer.write_record(["x", "z", "off", "y"]).expect("write header");
    for _ in 0..rows {
        let x = normal();
        let z = 0.5 * x + normal();
        let off = 0.25 * x;
        let y = -0.2 + 0.5 * x + 0.6 * z + normal() > 0.0;
        writer
            .write_record([
                format!("{x:.17e}"),
                format!("{z:.17e}"),
                format!("{off:.17e}"),
                u8::from(y).to_string(),
            ])
            .expect("write fixture row");
    }
    writer.flush().expect("flush fixture");
}

fn gam(scratch: &Path, args: &[&str]) -> Output {
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

/// True when some object in the document maps `key` to the string `value`.
fn records_string(document: &serde_json::Value, key: &str, value: &str) -> bool {
    match document {
        serde_json::Value::Object(map) => map.iter().any(|(name, child)| {
            (name == key && child.as_str() == Some(value)) || records_string(child, key, value)
        }),
        serde_json::Value::Array(items) => items.iter().any(|child| records_string(child, key, value)),
        _ => false,
    }
}

#[test]
fn bms_noise_offset_column_is_the_slope_offset_on_the_cli() {
    let scratch = tempfile::tempdir().expect("scratch dir");
    let dir = scratch.path();
    write_fixture(&dir.join("train.csv"), 600, 4343);

    let fitted = gam(
        dir,
        &[
            "fit",
            "train.csv",
            "y ~ x",
            "--family",
            "bernoulli-marginal-slope",
            "--slope-formula",
            "1",
            "--z-column",
            "z",
            "--noise-offset-column",
            "off",
            "--out",
            "bms.gam",
        ],
    );
    assert!(
        fitted.status.success(),
        "a marginal-slope fit takes --noise-offset-column as its slope offset: {}",
        describe(&fitted)
    );
    let saved = std::fs::read_to_string(dir.join("bms.gam")).expect("read the saved model");
    let document: serde_json::Value = serde_json::from_str(&saved).expect("the saved model is JSON");
    assert!(
        records_string(&document, "noise_offset_column", "off"),
        "the saved marginal-slope model records its slope offset column"
    );

    let refused = gam(
        dir,
        &["fit", "train.csv", "y ~ x", "--noise-offset-column", "off", "--out", "plain.gam"],
    );
    assert!(
        !refused.status.success(),
        "a fit with no noise formula and no marginal-slope predictor would drop the column: {}",
        describe(&refused)
    );
    let stderr = String::from_utf8_lossy(&refused.stderr);
    assert!(
        stderr.contains("noise_offset_column requires a location-scale model"),
        "the refusal names what the column needs: {stderr}"
    );
    assert!(!dir.join("plain.gam").exists(), "a refused fit writes no model");
}
