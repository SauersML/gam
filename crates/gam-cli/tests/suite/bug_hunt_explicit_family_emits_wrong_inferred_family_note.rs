//! Bug hunt (#1781): `gam fit` prints a factually wrong family-inference note
//! whenever the user passes an explicit `--family` but leaves `link(type=...)`
//! at its default. For example
//!
//!   gam fit gamma.csv "y ~ s(x)" --family gamma-log
//!
//! prints on stderr:
//!
//!   "- Inferred gaussian-identity family for response 'y' because values are
//!    not strictly binary. Override with link(type=...)."
//!
//! even though the fitted+saved model is Gamma/log. The note asserts a data-based
//! inference that never happened and names the wrong family.
//!
//! ROOT CAUSE: the inference note in `run_fit.rs` was gated ONLY on
//! `link_choice.is_none()`, ignoring whether the family was actually
//! auto-inferred. When `--family` is explicit, `resolve_family` uses the
//! requested family and no inference occurs, so no note should be emitted. The
//! note must additionally require `matches!(args.family, FamilyArg::Auto)`.
//!
//! This test fits a strictly-positive continuous response with an explicit
//! `--family gamma-log` (default link) and asserts stderr does NOT contain the
//! bogus "Inferred gaussian-identity family" note. Before the fix the note is
//! emitted; after the fix it is not.
//!
//! # The fixture must carry real dispersion — do not "tidy" it back to a curve
//!
//! `bug_hunt_explicit_family_gamma.csv` used to be `y = exp(0.3x + 0.5)`
//! evaluated exactly on a regular grid: deterministic, zero residual
//! dispersion, `y` strictly increasing across all 40 rows. A Gamma-distributed
//! response cannot be monotone in 40 consecutive draws — that was a noiseless
//! curve wearing a Gamma label.
//!
//! `s(x)` fits such data to essentially zero residual, and the Gamma shape MLE
//! for zero dispersion is `+inf`: the dispersion statistic sits inside its own
//! rounding band, so `pirls::dispersion` refuses with
//!
//!   Gamma shape MLE is not finite: the dispersion statistic ... is inside its rounding band
//!
//! and the whole fit dies before reaching the line this test is about. **That
//! refusal is correct behaviour, not a defect** — the fixture was asking the
//! estimator for a quantity that does not exist on this data (gam#2665).
//!
//! The response is now drawn from the Gamma it claims to be, keeping the same
//! mean curve and the same `x` grid:
//!
//!   y_i ~ Gamma(shape = 4, scale = mu_i / 4),  mu_i = exp(0.3 x_i + 0.5)
//!
//! numpy `default_rng(20260801)`, values rounded to 6 decimals, so the
//! committed file is reproducible. Shape 4 is a coefficient of variation of
//! `1/sqrt(4) = 0.5`; the profile target measured on the true mean is
//! `1.2407e-1`, which recovers a shape of `4.03`. The response now rises on
//! 21 of 39 consecutive steps rather than 39 of 39.
//!
//! Nothing here needs a well-conditioned Gamma fit — the assertion is about a
//! stderr note — but the fit does have to REACH the point where the note would
//! be printed, and a zero-dispersion response never gets there.

use std::process::Command;

#[test]
fn explicit_family_does_not_emit_wrong_inferred_family_note() {
    let fixture = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/bug_hunt_explicit_family_gamma.csv"
    );
    let out = tempfile::Builder::new()
        .suffix(".gam")
        .tempfile()
        .expect("temp output path");

    let output = Command::new(gam_test_support::gam_binary!())
        .arg("fit")
        .arg(fixture)
        .arg("y ~ s(x)")
        .arg("--family")
        .arg("gamma-log")
        .arg("--out")
        .arg(out.path())
        .output()
        .expect("spawn gam fit");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    // The precise defect: an explicit non-gaussian `--family` still triggers the
    // data-inference note, which both claims an inference that never happened and
    // names the wrong (gaussian) family for the actually-fitted Gamma model.
    assert!(
        !stderr.contains("Inferred gaussian-identity family"),
        "explicit `--family gamma-log` emitted the bogus gaussian-identity \
         inference note; the note must be gated on the family actually being \
         auto-inferred (FamilyArg::Auto), not merely on the link being default.\n\
         stderr tail: {}",
        stderr.lines().rev().take(8).collect::<Vec<_>>().join("\n")
    );

    // Sanity: the explicit-family fit itself must succeed through the CLI.
    assert!(
        output.status.success(),
        "gam fit --family gamma-log failed (exit {:?}).\nstdout tail: {}\nstderr tail: {}",
        output.status.code(),
        stdout.lines().rev().take(6).collect::<Vec<_>>().join("\n"),
        stderr.lines().rev().take(6).collect::<Vec<_>>().join("\n")
    );
}
