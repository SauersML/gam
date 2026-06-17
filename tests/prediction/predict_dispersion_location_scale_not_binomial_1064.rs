//! Regression for #1064: the predict path mis-classified dispersion
//! location-scale (#913) models as binomial-location-scale.
//!
//! A fitted Gamma / NegativeBinomial / Beta / Tweedie dispersion model
//! (`--family <fam>-log --predict-noise <formula>`) is a `ModelKind::LocationScale`
//! with a *non-Gaussian* likelihood. Before the fix, `FittedModel::from_payload`
//! / `predict_model_class` classified **any** non-Gaussian location-scale
//! payload as `BinomialLocationScale`, so the saved dispersion model was
//! predicted with the binomial threshold-scale predictor
//! (`binomial_location_scale_threshold_beta` + a scale-deviation
//! `noise_transform` it never carries). The result was either a hard failure or
//! predictions squashed onto the binomial probability scale `[0, 1]`.
//!
//! After the fix the dispersion class routes through the GLM mean inverse link
//! (log for Gamma/NB/Tweedie). This test fits a Gamma dispersion model whose
//! true mean `mu = exp(0.5 + 0.8 x)` ranges well above 1, predicts on a held-out
//! grid, and asserts the predicted means are on the *gamma* scale — strictly
//! positive, monotone in `x`, and far outside the `[0, 1]` band a binomial
//! threshold predictor would produce.

use gam::test_support::cli_harness::{
    read_prediction_means, run_capture_or_panic, run_or_panic, write_predict_csv_rows,
};
use std::path::Path;
use std::process::Command;

/// True mean surface: log-link `mu(x) = exp(0.5 + 0.8 x)` on `x in [-1, 1]`,
/// so `mu` runs from ~0.74 (x = -1) to ~3.67 (x = 1) — well above the binomial
/// `[0, 1]` band at the upper end.
fn true_mean(x: f64) -> f64 {
    (0.5 + 0.8 * x).exp()
}

fn write_training_csv(path: &Path) {
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use rand_distr::{Distribution, Gamma};
    let mut rng = StdRng::seed_from_u64(1064);
    let mut writer = csv::Writer::from_path(path).expect("create training csv");
    writer.write_record(["x", "y"]).expect("write header");
    let n = 400usize;
    for i in 0..n {
        let x = -1.0 + 2.0 * (i as f64) / ((n - 1) as f64);
        let mu = true_mean(x);
        // Overdispersion that varies with x so the noise channel carries signal;
        // shape `k = exp(1.5 - x)` (precision), scale `mu / k` => mean `mu`.
        let k = (1.5 - x).exp();
        let gamma = Gamma::new(k, mu / k).expect("gamma");
        let y = gamma.sample(&mut rng).max(1e-6);
        writer
            .write_record([format!("{x:.10}"), format!("{y:.10}")])
            .expect("write training row");
    }
    writer.flush().expect("flush training csv");
}

fn fit_gamma_dispersion(train_path: &Path, model_path: &Path) -> String {
    let mut fit_cmd = Command::new(gam::gam_binary!());
    fit_cmd
        .arg("fit")
        .arg(train_path)
        .arg("y ~ x")
        .args(["--family", "gamma-log"])
        // The noise (dispersion) formula is what routes `--family gamma-log`
        // to the dispersion location-scale family (#913).
        .args(["--predict-noise", "x"])
        .arg("--out")
        .arg(model_path);
    let label = "gam fit gamma dispersion location-scale";
    let stdout = run_capture_or_panic(fit_cmd, label);
    assert!(model_path.is_file(), "{label} did not write {model_path:?}");
    stdout
}

#[test]
fn dispersion_location_scale_predict_is_gamma_scale_not_binomial() {
    let dir = tempfile::tempdir().expect("tempdir");
    let train_path = dir.path().join("train.csv");
    let model_path = dir.path().join("model.gam");
    let predict_path = dir.path().join("newdata.csv");
    let out_path = dir.path().join("pred.csv");

    write_training_csv(&train_path);
    fit_gamma_dispersion(&train_path, &model_path);

    // Held-out grid spanning the training range; sorted ascending in x so the
    // predicted means must inherit the log-link monotonicity.
    let xs: Vec<f64> = (0..21).map(|i| -1.0 + 0.1 * (i as f64)).collect();
    write_predict_csv_rows(
        &predict_path,
        ["x"],
        xs.iter().map(|x| [format!("{x:.10}")]),
    );

    let mut predict_cmd = Command::new(gam::gam_binary!());
    predict_cmd
        .arg("predict")
        .arg(&model_path)
        .arg(&predict_path)
        .arg("--out")
        .arg(&out_path);
    run_or_panic(predict_cmd, "gam predict dispersion location-scale");

    let means = read_prediction_means(&out_path);
    assert_eq!(
        means.len(),
        xs.len(),
        "predicted means count must match the prediction grid"
    );

    // (1) Every predicted mean is strictly positive — the gamma response scale.
    for (x, m) in xs.iter().zip(means.iter()) {
        assert!(
            m.is_finite() && *m > 0.0,
            "gamma dispersion predicted mean at x={x} must be a positive response, got {m}"
        );
    }

    // (2) The upper-tail prediction is far above the binomial `[0, 1]` band: a
    //     binomial threshold-scale predictor (the #1064 mis-route) can only ever
    //     emit probabilities <= 1, so a mean comfortably above 1 proves the
    //     gamma log-link path is in force. True mu(1) ~ 3.67.
    let m_high = *means.last().expect("at least one prediction");
    assert!(
        m_high > 1.5,
        "predicted mean at x=1 ({m_high}) must be on the gamma scale (true mu~3.67), \
         not squashed into the binomial [0,1] band by the mis-routed threshold predictor"
    );

    // (3) The log-link mean is monotone increasing in x, so predictions track
    //     the exp(0.5 + 0.8 x) shape rather than a flat / inverted binomial
    //     surface. Allow tiny non-monotone noise from the smoother but require a
    //     clear overall increase.
    let m_low = means[0];
    assert!(
        m_high > m_low * 2.0,
        "log-link mean must increase across x (true ratio mu(1)/mu(-1) ~ 4.95): \
         got m_low={m_low}, m_high={m_high}"
    );

    // (4) The predicted means recover the truth: each within 35% of the true
    //     gamma mean on the held-out grid. A binomial mis-route could never
    //     match exp(0.5 + 0.8 x) at the upper end.
    for (x, m) in xs.iter().zip(means.iter()) {
        let truth = true_mean(*x);
        let rel = (m - truth).abs() / truth;
        assert!(
            rel < 0.35,
            "gamma dispersion mean at x={x} ({m}) should recover true mu={truth} \
             (relative error {rel:.3} >= 0.35)"
        );
    }
}
