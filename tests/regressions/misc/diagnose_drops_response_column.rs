//! Failing-ticket regression: `gam diagnose <model> <data>` cannot run on the
//! very data the model was fit on — it always aborts with
//!
//!   error: response column 'y' not found in data. Available columns: [x]
//!
//! even though the CSV plainly contains `y`. `diagnose` needs the response to
//! form leave-one-out residuals, but it loads its dataset through the
//! *prediction* column contract, which deliberately drops the response.
//!
//! Root cause (files/lines read):
//!
//!  * `run_diagnose` loads the diagnostic dataset with
//!    `load_datasetwith_model_schema(&args.data, &model)`
//!    (`src/main.rs:3897`). That helper projects the frame onto
//!    `FittedModel::prediction_required_columns()` (`src/main.rs:8162` →
//!    `..._extra` → `load_dataset_auto_with_schema_projected`), the post-#840
//!    contract that keeps only the columns the *formula* names so unrelated
//!    ID/label columns don't strict-validate at predict time.
//!
//!  * `prediction_required_columns()` (`src/inference/model.rs:2484`) collects
//!    only RHS term columns plus offsets, and adds the *response* only for
//!    `Surv(...)` responses and `TransformationNormal` models — never the bare
//!    response of a standard GAM. So for `y ~ s(x)` the projected frame is just
//!    `{x}`; `y` is dropped.
//!
//!  * `run_diagnose` then immediately needs the response:
//!    `resolve_role_col(&col_map, &parsed.response, "response")`
//!    (`src/main.rs:3903`) fails because `y` is no longer in the column map,
//!    raising the "response column 'y' not found" error from
//!    `src/inference/data.rs:131`.
//!
//!  * The two model classes whose response *is* kept by
//!    `prediction_required_columns()` (Surv / TransformationNormal) are
//!    explicitly rejected by `diagnose` up front (`src/main.rs:3888`,
//!    `predict_model_class() != Standard`). So `diagnose` is unusable for
//!    exactly the model class it claims to support: every standard GAM fit.
//!
//! Expected: `gam diagnose model.json train.csv` succeeds and prints its ALO
//! diagnostics table (the success path ends with `cli_out!("ALO diagnostics
//! (top leverage rows):")`, `src/main.rs:4026`).
//!
//! Observed: it aborts before computing anything, on the data the model was
//! just fit on.
//!
//! The fix is to load the response for `diagnose` (e.g. request the response
//! column as an `extra_required` for the projected loader, or use a
//! non-projected load). This test drives the real `gam` binary end to end: it
//! fails today at the `diagnose` step and passes once the response column is
//! available, with no edits.
//!
//! NOTE: building the `gam` binary this test spawns also requires the crate to
//! compile, which it currently does not — see the sibling
//! `bug_hunt_sphere_gpu_macro_unscoped_breaks_build` ticket. Once that
//! one-line compile fix lands, this test builds and exercises the diagnose bug.

use std::path::Path;
use std::process::Command;

fn write_training_csv(path: &Path) {
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use rand_distr::{Distribution, Normal};
    let mut rng = StdRng::seed_from_u64(7);
    let noise = Normal::new(0.0, 0.05).expect("normal");
    let mut writer = csv::Writer::from_path(path).expect("create training csv");
    writer.write_record(["y", "x"]).expect("write header");
    let n = 300usize;
    for i in 0..n {
        let x = (i as f64) / ((n - 1) as f64);
        let y = (2.0 * std::f64::consts::PI * x).sin() + noise.sample(&mut rng);
        writer
            .write_record([format!("{y:.10}"), format!("{x:.10}")])
            .expect("write training row");
    }
    writer.flush().expect("flush training csv");
}

fn fit_model(train_path: &Path, model_path: &Path) {
    let out = Command::new(gam::gam_binary!())
        .arg("fit")
        .arg(train_path)
        .arg("y ~ s(x)")
        .args(["--family", "gaussian"])
        .arg("--out")
        .arg(model_path)
        .output()
        .expect("spawn gam fit");
    assert!(
        out.status.success(),
        "`gam fit 'y ~ s(x)'` failed.\n--- stderr ---\n{}",
        String::from_utf8_lossy(&out.stderr),
    );
    assert!(model_path.is_file(), "gam fit did not write {model_path:?}");
}

/// Survival analogue of the response-drop bug (#2301): a `Surv(time, event)`
/// fit's EVENT column is not prediction-required (prediction never reads whether
/// the event occurred), so the projected diagnostics loader dropped it and
/// `gam diagnose --alo` aborted with "survival event column 'event' not found".
///
/// This exercises the *2-argument* right-censored `Surv(time, event)` form —
/// deliberately different from the committed `bug_hunt_2301` 3-argument
/// `Surv(entry, exit, event)` arm — so a regression that only re-added the event
/// column for one Surv arity would still be caught here. The `event` column must
/// survive the projection and ALO must print its typed survival coordinate frame.
fn write_survival_csv(path: &Path) {
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use rand_distr::{Distribution, Uniform};
    let mut rng = StdRng::seed_from_u64(2301);
    let unit = Uniform::new(0.0_f64, 1.0_f64).expect("uniform");
    let mut writer = csv::Writer::from_path(path).expect("create survival csv");
    writer
        .write_record(["time", "event", "x"])
        .expect("write header");
    let n = 200usize;
    for _ in 0..n {
        let x = -1.5 + 3.0 * unit.sample(&mut rng);
        let u = unit.sample(&mut rng).clamp(1e-9, 1.0 - 1e-12);
        let scale = (0.5 + 0.4 * x).exp();
        let t_latent = scale * (-u.ln());
        let censor = 4.0_f64;
        let exit = t_latent.min(censor);
        let event = if t_latent <= censor { 1.0 } else { 0.0 };
        writer
            .write_record([format!("{exit:.10}"), format!("{event:.1}"), format!("{x:.10}")])
            .expect("write survival row");
    }
    writer.flush().expect("flush survival csv");
}

#[test]
fn diagnose_alo_keeps_survival_event_column() {
    let dir = tempfile::tempdir().expect("create tempdir");
    let train_path = dir.path().join("surv_train.csv");
    let model_path = dir.path().join("surv_model.gam");

    write_survival_csv(&train_path);

    let fit = Command::new(gam::gam_binary!())
        .arg("fit")
        .arg(&train_path)
        .arg("Surv(time, event) ~ x")
        .args(["--survival-likelihood", "weibull"])
        .arg("--out")
        .arg(&model_path)
        .output()
        .expect("spawn gam fit (survival)");
    assert!(
        fit.status.success(),
        "`gam fit 'Surv(time, event) ~ x'` failed.\n--- stderr ---\n{}",
        String::from_utf8_lossy(&fit.stderr),
    );
    assert!(model_path.is_file(), "gam fit did not write {model_path:?}");

    let out = Command::new(gam::gam_binary!())
        .arg("diagnose")
        .arg("--alo")
        .arg(&model_path)
        .arg(&train_path)
        .output()
        .expect("spawn gam diagnose --alo (survival)");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);

    assert!(
        !stderr.contains("survival event column") && !stdout.contains("survival event column"),
        "`gam diagnose --alo` dropped the survival event column from its own \
         training data:\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}"
    );
    assert!(
        out.status.success(),
        "`gam diagnose --alo` failed on a survival fit's training data.\n\
         --- stdout ---\n{stdout}\n--- stderr ---\n{stderr}"
    );
    assert!(
        stdout.contains("ALO diagnostics"),
        "survival `gam diagnose --alo` produced no ALO diagnostics table.\n\
         --- stdout ---\n{stdout}\n--- stderr ---\n{stderr}"
    );
    assert!(
        stdout.contains("ALO coordinates (survival)"),
        "survival `gam diagnose --alo` did not print its typed survival ALO frame.\n\
         --- stdout ---\n{stdout}"
    );
}

#[test]
fn diagnose_runs_on_the_training_data() {
    let dir = tempfile::tempdir().expect("create tempdir");
    let train_path = dir.path().join("train.csv");
    let model_path = dir.path().join("model.json");

    write_training_csv(&train_path);
    fit_model(&train_path, &model_path);

    // Diagnose the model on the *exact* data it was fit on. This must work: the
    // file contains the response column `y` the model needs for leave-one-out
    // diagnostics.
    let out = Command::new(gam::gam_binary!())
        .arg("diagnose")
        .arg(&model_path)
        .arg(&train_path)
        .output()
        .expect("spawn gam diagnose");

    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);

    assert!(
        !stderr.contains("response column") && !stdout.contains("response column"),
        "`gam diagnose` dropped the response column from the data it was fit on:\n\
         --- stdout ---\n{stdout}\n--- stderr ---\n{stderr}"
    );
    assert!(
        out.status.success(),
        "`gam diagnose model train.csv` failed on the training data.\n\
         --- stdout ---\n{stdout}\n--- stderr ---\n{stderr}"
    );
    assert!(
        stdout.contains("ALO diagnostics") || stderr.contains("ALO diagnostics"),
        "`gam diagnose` exited 0 but produced no ALO diagnostics table.\n\
         --- stdout ---\n{stdout}\n--- stderr ---\n{stderr}"
    );
}
