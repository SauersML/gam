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

    let output = Command::new(env!("CARGO_BIN_EXE_gam"))
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
