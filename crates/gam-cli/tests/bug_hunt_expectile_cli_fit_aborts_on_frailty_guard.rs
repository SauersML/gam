//! Bug hunt (#1780): `gam fit --family expectile` (Newey–Powell LAWS expectile
//! regression) aborted at fit time on every dataset with
//!
//!   "config.frailty is not supported for standard family
//!    LikelihoodSpec { response: Gaussian, link: Standard(Identity) };
//!    use a frailty-aware family instead"
//!
//! ROOT CAUSE — sentinel conflation of `Some(FrailtySpec::None)` with a real
//! frailty request. The CLI config layer ALWAYS populates the frailty config
//! (`resolve_cli_fit_config` sets `fit_config.frailty = Some(resolve_cli_frailty_spec(..))`,
//! which returns `FrailtySpec::None` when no `--frailty-kind` is passed), so
//! every ordinary CLI fit carries `frailty == Some(FrailtySpec::None)` — the
//! canonical "no frailty" value, but a `Some(..)`. The expectile driver clones
//! that config into its inner Gaussian design config, and the standard
//! materializer's frailty guard was written as `config.frailty.is_some()`, which
//! fires on the null `Some(FrailtySpec::None)` and rejects the fit. The library
//! default (`FitConfig::default().frailty == None`) dodged it, which is why this
//! was uniquely a CLI-config manifestation.
//!
//! FIX — the guards now test `FrailtySpec::is_active` (a real, non-`None`
//! frailty) instead of `Option::is_some`, so the null `Some(FrailtySpec::None)`
//! no longer masquerades as a frailty request.
//!
//! This test fits a deterministic continuous response through the real `gam fit`
//! CLI with `--family expectile` and asserts the fit SUCCEEDS and never trips the
//! frailty guard.

use std::process::Command;

#[test]
fn expectile_cli_fit_does_not_abort_on_null_frailty_guard() {
    let fixture = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/bug_hunt_expectile_laws_cli.csv"
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

    assert!(
        !combined.contains("config.frailty is not supported"),
        "expectile fit aborted on the standard-family frailty guard — the CLI's \
         null Some(FrailtySpec::None) leaked into the inner Gaussian config and \
         the `is_some()` guard misread it as a frailty request.\nstderr tail: {}",
        stderr.lines().rev().take(6).collect::<Vec<_>>().join("\n")
    );

    assert!(
        output.status.success(),
        "gam fit --family expectile failed (exit {:?}).\nstderr tail: {}",
        output.status.code(),
        stderr.lines().rev().take(8).collect::<Vec<_>>().join("\n")
    );
}
