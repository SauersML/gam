//! The command-line fit surface must expose the family-owned all-zero count
//! refusal, not enter PIRLS and leak an optimizer-internal error (#2255).

use std::process::Command;

#[test]
fn cli_refuses_all_zero_poisson_and_negative_binomial_before_optimization() {
    let fixture = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/bug_hunt_all_zero_count_2255.csv"
    );

    for family in ["poisson-log", "negative-binomial"] {
        let output_path = tempfile::Builder::new()
            .suffix(".gam")
            .tempfile()
            .expect("temporary output path");
        let output = Command::new(env!("CARGO_BIN_EXE_gam"))
            .arg("fit")
            .arg(fixture)
            .arg("y ~ 1")
            .arg("--family")
            .arg(family)
            .arg("--out")
            .arg(output_path.path())
            .output()
            .expect("spawn gam fit");
        let stderr = String::from_utf8_lossy(&output.stderr);

        assert!(
            !output.status.success(),
            "{family} all-zero response unexpectedly minted a fit"
        );
        assert!(
            stderr.contains("all counts are 0"),
            "{family} did not expose the family degeneracy: {stderr}"
        );
        assert!(
            stderr.contains("no finite fitted mode or finite posterior mean/variance"),
            "{family} error did not explain the unbounded estimand: {stderr}"
        );
        assert!(
            stderr.contains("at least one positive count"),
            "{family} error was not actionable: {stderr}"
        );
        assert!(
            !stderr.contains("StalledAtValidMinimum")
                && !stderr.contains("IntegrationError")
                && !stderr.contains("NegativeBinomialAlternationDidNotConverge"),
            "{family} reached optimization instead of the validation boundary: {stderr}"
        );
    }
}
