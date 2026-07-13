//! Bug hunt (#1780): `gam fit --family expectile` (Newey–Powell LAWS expectile
//! regression) routes to the LAWS driver but ABORTS at fit time on every dataset
//! with:
//!
//!   "expectile fit failed: config.frailty is not supported for standard family
//!    LikelihoodSpec { response: Gaussian, link: Standard(Identity) }; use a
//!    frailty-aware family instead"
//!
//! ROOT CAUSE (confirmed):
//!   1. The CLI always populates the frailty config even with no `--frailty-kind`
//!      (gam-config/src/lib.rs): `fit_config.frailty = resolve_cli_frailty_spec(..)`,
//!      which returns `FrailtySpec::None` — so `frailty == Some(FrailtySpec::None)`
//!      (semantically "no frailty", but a `Some`).
//!   2. The expectile driver clones that config to build its inner Gaussian config
//!      but did NOT clear `frailty`
//!      (gam-models/src/fit_orchestration/entry.rs, `fit_expectile_laws`).
//!   3. The standard materializer's frailty guard rejects ANY `Some(..)`
//!      (gam-models/src/fit_orchestration/materialize/standard.rs).
//!
//! The inner Gaussian-identity design genuinely has no frailty, so the fix clears
//! `frailty` on the inner `gaussian_config`. This test fits a `y ~ s(x)` expectile
//! model through the real `gam fit` CLI and asserts the fit SUCCEEDS. Before the
//! fix it aborts with the frailty-guard error above.

use std::process::Command;

#[test]
fn expectile_cli_fit_does_not_abort_on_standard_frailty_guard() {
    let fixture = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/bug_hunt_expectile_frailty_guard.csv"
    );
    let out = tempfile::Builder::new()
        .suffix(".gam")
        .tempfile()
        .expect("temp output path");

    let output = Command::new(env!("CARGO_BIN_EXE_gam"))
        .arg("fit")
        .arg(fixture)
        .arg("y ~ s(x)")
        .arg("--family")
        .arg("expectile")
        .arg("--expectile-tau")
        .arg("0.5")
        .arg("--out")
        .arg(out.path())
        .output()
        .expect("spawn gam fit");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let combined = format!("{stdout}\n{stderr}");

    // The precise defect: the inner Gaussian config carried
    // `frailty = Some(FrailtySpec::None)` into the standard materializer, whose
    // guard rejects any `Some(..)`.
    assert!(
        !combined.contains("config.frailty is not supported"),
        "expectile fit aborted on the standard-family frailty guard — the inner \
         Gaussian config in fit_expectile_laws carried `frailty = Some(FrailtySpec::None)` \
         instead of `None`.\nstderr tail: {}",
        stderr.lines().rev().take(6).collect::<Vec<_>>().join("\n")
    );
    assert!(
        !combined.contains("expectile fit failed"),
        "expectile fit failed through the CLI.\nstderr tail: {}",
        stderr.lines().rev().take(6).collect::<Vec<_>>().join("\n")
    );

    // And the documented expectile family must actually fit through the CLI.
    assert!(
        output.status.success(),
        "gam fit --family expectile failed (exit {:?}).\nstderr tail: {}",
        output.status.code(),
        stderr.lines().rev().take(6).collect::<Vec<_>>().join("\n")
    );

    let saved: serde_json::Value =
        serde_json::from_slice(&std::fs::read(out.path()).expect("read saved expectile model"))
            .expect("decode saved expectile model");
    assert_eq!(
        saved["payload"]["estimator"],
        serde_json::json!({"estimator_kind": "expectile", "tau": 0.5}),
        "CLI persistence must retain the expectile target instead of erasing it into Gaussian"
    );

    let generated = tempfile::Builder::new()
        .suffix(".csv")
        .tempfile()
        .expect("temp generated output path");
    let generation = Command::new(env!("CARGO_BIN_EXE_gam"))
        .arg("generate")
        .arg(out.path())
        .arg(fixture)
        .arg("--n-draws")
        .arg("2")
        .arg("--out")
        .arg(generated.path())
        .output()
        .expect("spawn gam generate");
    let generation_error = String::from_utf8_lossy(&generation.stderr);
    assert!(
        !generation.status.success()
            && generation_error.contains("expectile")
            && generation_error.contains("no observation-replicate sampler"),
        "an expectile target has no implied observation law and generation must refuse it; \
         status={:?}, stderr={generation_error}",
        generation.status.code(),
    );
}
