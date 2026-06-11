//! Right-censored survival shorthand (`Surv(time, event)`) must be a
//! first-class input across the whole `gam fit` → `gam predict` → `gam
//! sample` pipeline. `gam fit` and `gam predict` both accept it today —
//! `fit` synthesizes an implicit zero-entry column and `predict`'s CLI
//! path (`src/main.rs::run_predict_survival` around line 3294) reads
//! `survival_entry` as `Option<&str>` and falls back to `entry_val = 0.0`
//! when it is `None`. The posterior-sample path does not.
//!
//! `src/inference/sample.rs::sample_survival` (around line 648) treats
//! `survival_entry` as mandatory via `.ok_or_else(...)`, so for any model
//! trained with the shorthand it bails out with
//! `"survival model missing entry column metadata"` before any draws are
//! taken — even though the model is otherwise complete and `gam predict`
//! on the same artifact returns finite predictions. The same
//! `ok_or_else` pattern is also baked into the library functions in
//! `src/families/survival_predict.rs:206,638`, so a Python `model.predict`
//! through `gam-pyffi` (`crates/gam-pyffi/src/lib.rs:29305`) hits the
//! same dead-end on shorthand-trained survival models.
//!
//! This integration test exercises the end-to-end CLI workflow: write a
//! small right-censored survival dataset, fit it with the shorthand,
//! sanity-check that `gam predict` reads the model, then assert that
//! `gam sample` does not reject the model with the shorthand-specific
//! error string. When `survival_entry == None` is handled symmetrically
//! across all three commands the test passes; today it fails because
//! `gam sample` short-circuits with the entry-metadata error.

use gam::test_support::cli_harness::run_capture_or_panic;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Exp, Uniform};
use std::path::Path;
use std::process::Command;

const N: usize = 200;
const SEED: u64 = 20260528;

/// Write a tiny right-censored survival CSV with three columns: `t`
/// (positive exit time), `event` (0/1 censoring indicator), and `x`
/// (continuous covariate). No `entry` column is emitted — the test is
/// specifically about the `Surv(t, event)` shorthand. The frequency
/// and ranges are chosen so a transformation-normal survival fit
/// converges deterministically in a few PIRLS iterations on a stock
/// CI box.
fn write_shorthand_survival_fixture(csv_path: &Path) {
    let mut rng = StdRng::seed_from_u64(SEED);
    let cov_dist = Uniform::new(-1.0_f64, 1.0_f64).expect("uniform [-1, 1)");
    let baseline_failure_dist = Exp::new(1.0_f64).expect("exp(1.0) rate");
    let baseline_censor_dist = Exp::new(0.3_f64).expect("censoring exp rate");

    let mut writer = csv::Writer::from_path(csv_path).expect("create survival fixture csv");
    writer
        .write_record(["t", "event", "x"])
        .expect("write csv header");
    for _ in 0..N {
        let x = cov_dist.sample(&mut rng);
        // Hazard ratio exp(0.6·x): bigger x ⇒ faster failure. The
        // exact shape doesn't matter for the bug under test (we only
        // need the model to fit), but a real covariate effect avoids
        // identifiability surprises in the smoothing optimizer.
        let hazard_scale = (-0.6 * x).exp();
        let failure_time = baseline_failure_dist.sample(&mut rng) * hazard_scale;
        let censor_time = baseline_censor_dist.sample(&mut rng);
        let event = if failure_time <= censor_time {
            1u8
        } else {
            0u8
        };
        let observed = failure_time.min(censor_time).max(1e-3);
        writer
            .write_record([
                format!("{observed:.10}"),
                event.to_string(),
                format!("{x:.10}"),
            ])
            .expect("write csv row");
    }
    writer.flush().expect("flush csv");
}

#[test]
fn gam_sample_succeeds_on_right_censored_shorthand_survival_model() {
    let dir = tempfile::tempdir().expect("create tempdir");
    let csv_path = dir.path().join("surv_shorthand.csv");
    write_shorthand_survival_fixture(&csv_path);

    let model_path = dir.path().join("surv_shorthand.model.json");
    // `Surv(t, event)` — no entry column, so the saved model serializes
    // `survival_entry = None`. That asymmetry is the root of the bug.
    let formula = "Surv(t, event) ~ x";
    let mut fit_cmd = Command::new(gam::gam_binary!());
    fit_cmd
        .arg("fit")
        .arg(&csv_path)
        .arg(formula)
        .args(["--survival-likelihood", "transformation"])
        .arg("--out")
        .arg(&model_path);
    run_capture_or_panic(fit_cmd, "gam fit Surv(t, event) ~ x");
    assert!(
        model_path.is_file(),
        "expected `gam fit` to produce {model_path:?} for the right-censored \
         shorthand, but the file is missing"
    );

    // Cross-check that `gam predict` reads the same shorthand model
    // successfully. `run_predict_survival` already maps an absent
    // `survival_entry` to a synthesized zero-entry column, so this
    // command is expected to succeed both before and after the
    // shorthand fix lands. If this assertion ever fails the
    // sample-side test below would be a false positive: keep them
    // paired so the regression is unambiguous.
    let pred_path = dir.path().join("surv_shorthand.pred.csv");
    let mut predict_cmd = Command::new(gam::gam_binary!());
    predict_cmd
        .arg("predict")
        .arg(&model_path)
        .arg(&csv_path)
        .arg("--out")
        .arg(&pred_path);
    run_capture_or_panic(predict_cmd, "gam predict (shorthand survival)");
    assert!(
        pred_path.is_file(),
        "`gam predict` did not write {pred_path:?} for the shorthand model"
    );

    // The bug: `gam sample` rejects the same model with
    //     "survival model missing entry column metadata"
    // even though the artifact is otherwise complete. When the
    // sample-side code path (src/inference/sample.rs around line 648
    // and src/families/survival_predict.rs around lines 206/638) is
    // taught to synthesize a zero-entry column the way
    // `run_predict_survival` does, this command should succeed and
    // write at least one row of posterior draws.
    let sample_path = dir.path().join("surv_shorthand.sample.csv");
    let mut sample_cmd = Command::new(gam::gam_binary!());
    sample_cmd
        .arg("sample")
        .arg(&model_path)
        .arg(&csv_path)
        .args(["--chains", "1", "--samples", "4", "--warmup", "4"])
        .arg("--out")
        .arg(&sample_path);
    let output = sample_cmd
        .output()
        .unwrap_or_else(|err| panic!("failed to spawn `gam sample`: {err}"));
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    let combined = format!("{stdout}{stderr}");

    // Targeted assertion: the shorthand-specific failure must not be
    // surfaced. Other failure modes (e.g. NUTS convergence diagnostics)
    // would represent independent bugs and should be filed separately.
    assert!(
        !combined.contains("survival model missing entry column metadata"),
        "`gam sample` rejected a right-censored shorthand model with the \
         shorthand-specific metadata error.\n\
         --- formula ---\n{formula}\n\
         --- stdout ---\n{stdout}\
         --- stderr ---\n{stderr}"
    );

    // And the command itself must succeed. After the fix the shorthand
    // model carries enough information to draw posterior samples.
    assert!(
        output.status.success(),
        "`gam sample` failed on a right-censored shorthand survival model \
         (exit status {})\n\
         --- stdout ---\n{stdout}\
         --- stderr ---\n{stderr}",
        output.status,
    );

    assert!(
        sample_path.is_file(),
        "`gam sample` reported success but did not write {sample_path:?}"
    );
    let posterior_rows = csv::Reader::from_path(&sample_path)
        .expect("open posterior csv")
        .records()
        .filter_map(Result::ok)
        .count();
    assert!(
        posterior_rows > 0,
        "`gam sample` wrote an empty posterior file at {sample_path:?}"
    );
}
