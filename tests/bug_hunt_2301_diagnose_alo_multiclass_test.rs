//! Regression for #2301: `gam diagnose --alo` covered only Standard-class
//! fits and rejected every other model class with a bare "unavailable in
//! this binary" message that named nothing and explained nothing.
//!
//! The fix generalized the saved-model ALO core
//! (`gam_predict::compute_saved_model_alo`) to dispatch location-scale
//! (Gaussian / binomial / dispersion), Bernoulli marginal-slope, and
//! transformation-normal fits through the shared parameter-aligned,
//! saved-Hessian ALO machinery, and `gam diagnose` now calls that one
//! dispatcher instead of duplicating a Standard-only replay. Survival is the
//! one class that genuinely cannot reuse this machinery (leave-one-out for a
//! risk-set likelihood needs typed event/time row replay, which does not
//! exist yet), so it keeps a hard error — but the error must name the class
//! and explain why, not just say "unavailable".
//!
//! This test drives the real CLI (`gam fit` + `gam diagnose --alo`) on:
//! 1. a Gaussian location-scale fit (`--predict-noise`) — must now SUCCEED
//!    and print an ALO diagnostics table;
//! 2. a survival fit (`Surv(...) ~ x`) — must still FAIL, but with a message
//!    that names "survival" and explains the risk-set/event-replay reason,
//!    not the old bare "unavailable in this binary" wording.

use std::path::Path;
use std::process::Command;

use gam::gam_binary;
use gam::test_support::cli_harness::run_or_panic;

fn write_csv(path: &Path, header: &[&str], rows: &[Vec<f64>]) {
    let mut writer = csv::Writer::from_path(path).expect("create csv");
    writer.write_record(header).expect("write header");
    for row in rows {
        let record: Vec<String> = row.iter().map(|value| format!("{value:.10}")).collect();
        writer.write_record(&record).expect("write row");
    }
    writer.flush().expect("flush csv");
}

#[test]
fn diagnose_alo_supports_gaussian_location_scale_and_names_survival_reason_2301() {
    let dir = tempfile::tempdir().expect("create tempdir");

    // --- Part 1: Gaussian location-scale fit gets real ALO diagnostics. ---
    let mut rng_state: u64 = 0x2301_2301_2301_2301;
    let mut next = || {
        // Small xorshift so this test has no external RNG crate dependency.
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        (rng_state >> 11) as f64 / (1u64 << 53) as f64
    };
    let n = 150;
    let mut ls_rows = Vec::with_capacity(n);
    for _ in 0..n {
        let x = -2.0 + 4.0 * next();
        let noise_scale = (0.2 + 0.3 * x).exp();
        // Box-Muller for an approximately Gaussian residual.
        let u1 = next().max(1e-12);
        let u2 = next();
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        let y = 1.0 + 2.0 * x + noise_scale * z;
        ls_rows.push(vec![y, x]);
    }
    let ls_data = dir.path().join("ls.csv");
    let ls_model = dir.path().join("ls.gam");
    write_csv(&ls_data, &["y", "x"], &ls_rows);

    let mut fit_ls = Command::new(gam_binary!());
    fit_ls
        .arg("fit")
        .args(["--family", "gaussian"])
        .arg("--predict-noise")
        .arg("x")
        .arg("--out")
        .arg(&ls_model)
        .arg(&ls_data)
        .arg("y ~ x");
    run_or_panic(fit_ls, "gam fit gaussian location-scale (#2301)");
    assert!(ls_model.is_file(), "gam fit did not write {ls_model:?}");

    let mut diagnose_ls = Command::new(gam_binary!());
    diagnose_ls
        .arg("diagnose")
        .arg("--alo")
        .arg(&ls_model)
        .arg(&ls_data);
    let ls_output = diagnose_ls
        .output()
        .expect("spawn gam diagnose --alo (location-scale)");
    let ls_stdout = String::from_utf8_lossy(&ls_output.stdout);
    let ls_stderr = String::from_utf8_lossy(&ls_output.stderr);
    assert!(
        ls_output.status.success(),
        "diagnose --alo must now succeed for a location-scale fit (#2301):\n\
         --- stdout ---\n{ls_stdout}\n--- stderr ---\n{ls_stderr}"
    );
    assert!(
        ls_stdout.contains("ALO diagnostics"),
        "location-scale diagnose must print an ALO table: {ls_stdout}"
    );
    assert!(
        ls_stdout.contains("ALO coordinates"),
        "location-scale diagnose must name its multicoordinate frame: {ls_stdout}"
    );

    // --- Part 2: survival fit still fails, but with a named, explained error. ---
    let mut surv_rows = Vec::with_capacity(n);
    for _ in 0..n {
        let x = -1.5 + 3.0 * next();
        let u = next().clamp(1e-9, 1.0 - 1e-12);
        let scale = (0.5 + 0.4 * x).exp();
        let t_latent = scale * (-u.ln());
        let censor = 4.0;
        let exit = t_latent.min(censor);
        let event = if t_latent <= censor { 1.0 } else { 0.0 };
        surv_rows.push(vec![0.0, exit, event, x]);
    }
    let surv_data = dir.path().join("surv.csv");
    let surv_model = dir.path().join("surv.gam");
    write_csv(&surv_data, &["entry", "exit", "event", "x"], &surv_rows);

    let mut fit_surv = Command::new(gam_binary!());
    fit_surv
        .arg("fit")
        .arg(&surv_data)
        .arg("Surv(entry, exit, event) ~ x")
        .args(["--survival-likelihood", "weibull"])
        .arg("--out")
        .arg(&surv_model);
    run_or_panic(fit_surv, "gam fit Weibull survival (#2301)");
    assert!(surv_model.is_file(), "gam fit did not write {surv_model:?}");

    let mut diagnose_surv = Command::new(gam_binary!());
    diagnose_surv
        .arg("diagnose")
        .arg("--alo")
        .arg(&surv_model)
        .arg(&surv_data);
    let surv_output = diagnose_surv
        .output()
        .expect("spawn gam diagnose --alo (survival)");
    let surv_stderr = String::from_utf8_lossy(&surv_output.stderr);
    assert!(
        !surv_output.status.success(),
        "survival ALO replay is not implemented yet and must still fail (#2301)"
    );
    assert!(
        surv_stderr.to_lowercase().contains("survival"),
        "survival diagnose error must name the class, not a bare 'unavailable': {surv_stderr}"
    );
    assert!(
        surv_stderr.contains("risk-set") || surv_stderr.contains("event/time"),
        "survival diagnose error must explain WHY (risk-set/event-time replay), not just fail silently: {surv_stderr}"
    );
    assert!(
        !surv_stderr.contains("unavailable in this binary"),
        "the old bare, unexplained error message must be gone: {surv_stderr}"
    );
}
