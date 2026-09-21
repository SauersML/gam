//! Regression (#4479): `gam predict` could never publish the observation
//! (prediction) band, and so a weighted fit's band could never be priced per
//! row.
//!
//! The shared predict policy (`gam_predict::interval_policy`) builds the band
//! `m ∓ z·√(σ̂²/w_i + se²)` that Python's `predict(observation_interval=True)`
//! returns (#2077). The CLI hardcoded the request off and forwarded no prior
//! weights. `gam predict --uncertainty --observation-interval` now publishes
//! `observation_lower` / `observation_upper` from that same policy.
//!
//! In the test, three query rows share one `x`, so they share the posterior
//! mean and its SE and differ only in the prior weight. With half-width
//! `h_w = z·√(σ̂²/w + se²)`, `(h_1² − h_2²)/(h_1² − h_4²) = (1 − 1/2)/(1 − 1/4)
//! = 2/3` for any `z`, `σ̂²` and `se`. The check therefore pins the weight law
//! itself, not a fitted number.

use std::path::Path;
use std::process::{Command, Output};

fn stderr_text(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

fn named_column(csv: &str, name: &str) -> Vec<f64> {
    let mut lines = csv.lines();
    let header = lines.next().expect("prediction CSV has a header row");
    let idx = header
        .split(',')
        .position(|h| h.trim() == name)
        .unwrap_or_else(|| panic!("prediction CSV has no `{name}` column; header: {header}"));
    lines
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            l.split(',')
                .nth(idx)
                .unwrap_or_else(|| panic!("row has a `{name}` cell"))
                .trim()
                .parse::<f64>()
                .unwrap_or_else(|err| panic!("`{name}` cell parses as f64: {err}"))
        })
        .collect()
}

fn predict(model: &Path, new_data: &Path, out: &Path, extra: &[&str]) -> Output {
    Command::new(gam_test_support::gam_binary!())
        .arg("predict")
        .arg(model)
        .arg(new_data)
        .arg("--out")
        .arg(out)
        .arg("--level")
        .arg("0.9")
        .args(extra)
        .output()
        .expect("spawn gam predict")
}

#[test]
fn predict_observation_interval_prices_each_row_from_its_prior_weight_4479() {
    let dir = tempfile::tempdir().expect("temp dir");
    let train = dir.path().join("train.csv");
    let new_data = dir.path().join("new.csv");
    let unweighted = dir.path().join("new_unweighted.csv");
    let model = dir.path().join("model.gam");
    let out = dir.path().join("pred.csv");

    let mut rows = String::from("x,y,w\n");
    for i in 0..48 {
        let x = i as f64 / 47.0;
        let y = 1.0 + 2.0 * x + 0.3 * (7.3 * i as f64).sin();
        let w = [1.0, 2.0, 4.0][i % 3];
        rows.push_str(&format!("{x},{y},{w}\n"));
    }
    std::fs::write(&train, rows).expect("write training csv");
    std::fs::write(&new_data, "x,w\n0.5,1.0\n0.5,2.0\n0.5,4.0\n").expect("write new csv");
    std::fs::write(&unweighted, "x\n0.5\n").expect("write unweighted csv");

    let fit = Command::new(gam_test_support::gam_binary!())
        .arg("fit")
        .arg(&train)
        .arg("y ~ x")
        .arg("--family")
        .arg("gaussian")
        .arg("--weights-column")
        .arg("w")
        .arg("--out")
        .arg(&model)
        .output()
        .expect("spawn gam fit");
    assert!(
        fit.status.success(),
        "weighted gaussian fit failed (exit {:?}): {}",
        fit.status.code(),
        stderr_text(&fit)
    );

    // The band is priced at `--level` beside the posterior band, never alone.
    let alone = predict(&model, &new_data, &out, &["--observation-interval"]);
    assert!(
        !alone.status.success() && stderr_text(&alone).contains("requires --uncertainty"),
        "--observation-interval without --uncertainty must be refused: {}",
        stderr_text(&alone)
    );
    // A weighted fit's band needs each new row's weight. It is never priced at
    // an unstated unit weight.
    let no_weight = predict(
        &model,
        &unweighted,
        &out,
        &["--uncertainty", "--observation-interval"],
    );
    assert!(
        !no_weight.status.success() && stderr_text(&no_weight).contains("'w'"),
        "a weighted fit's observation band must require the weight column `w`: {}",
        stderr_text(&no_weight)
    );

    let banded = predict(
        &model,
        &new_data,
        &out,
        &["--uncertainty", "--observation-interval"],
    );
    assert!(
        banded.status.success(),
        "gam predict --uncertainty --observation-interval failed (exit {:?}): {}",
        banded.status.code(),
        stderr_text(&banded)
    );
    let csv = std::fs::read_to_string(&out).expect("read predictions");
    let mean = named_column(&csv, "posterior_mean");
    let credible_lower = named_column(&csv, "posterior_mean_lower");
    let credible_upper = named_column(&csv, "posterior_mean_upper");
    let lower = named_column(&csv, "observation_lower");
    let upper = named_column(&csv, "observation_upper");
    assert_eq!(mean.len(), 3, "one prediction row per query row");

    let half: Vec<f64> = (0..3).map(|r| 0.5 * (upper[r] - lower[r])).collect();
    for r in 0..3 {
        assert!(
            (mean[r] - mean[0]).abs() <= 1e-10 * mean[0].abs().max(1.0),
            "rows at one x must share the posterior mean: {mean:?}"
        );
        let centre = 0.5 * (upper[r] + lower[r]);
        assert!(
            (centre - mean[r]).abs() <= 1e-9 * half[r],
            "a Gaussian observation band is centred on the mean: row {r} centre {centre} mean {}",
            mean[r]
        );
        let credible_half = 0.5 * (credible_upper[r] - credible_lower[r]);
        assert!(
            half[r] > credible_half,
            "the observation band must contain the credible band: row {r} {} vs {credible_half}",
            half[r]
        );
    }
    assert!(
        half[0] > half[1] && half[1] > half[2],
        "a heavier prior weight must narrow the observation band: {half:?}"
    );
    let ratio = (half[0].powi(2) - half[1].powi(2)) / (half[0].powi(2) - half[2].powi(2));
    assert!(
        (ratio - 2.0 / 3.0).abs() <= 1e-9,
        "the band must follow Var(y) = σ̂²/w + Var(μ): expected ratio 2/3, got {ratio}"
    );
}
