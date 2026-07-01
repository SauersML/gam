//! Bug hunt (#1781): `gam fit --family <non-gaussian>` printed a factually wrong
//! family-inference note whenever the user passed an explicit `--family` but left
//! the link at its default. Fitting positive-continuous data with
//! `--family gamma-log` emitted, on stderr:
//!
//!   "Inferred gaussian-identity family for response 'y' because values are not
//!    strictly binary. Override with link(type=...)."
//!
//! …while the model actually fit and saved was Gamma/log — nothing was inferred
//! (the user chose the family), and the named family was wrong.
//!
//! ROOT CAUSE — the note was gated ONLY on `link_choice.is_none()` (whether an
//! explicit `link(type=...)` was given), and its text was hard-coded from the
//! DATA heuristic, ignoring `args.family`. But the family was already resolved
//! from the explicit `--family` flag; the note describes the auto-discovery path
//! (`resolve_family` with `FamilyArg::Auto`), which did not run.
//!
//! FIX — the note is additionally gated on `matches!(args.family, FamilyArg::Auto)`,
//! so it is emitted only when the family was actually auto-inferred from the data.
//!
//! This test fits a deterministic positive-continuous response with an explicit
//! `--family gamma-log` and asserts NO "Inferred ... family" note is emitted.

use std::process::Command;

#[test]
fn explicit_family_does_not_emit_inferred_family_note() {
    let fixture = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/bug_hunt_explicit_family_note_cli.csv"
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
    let combined = format!("{stdout}\n{stderr}");

    assert!(
        output.status.success(),
        "gam fit --family gamma-log failed (exit {:?}).\nstderr tail: {}",
        output.status.code(),
        stderr.lines().rev().take(8).collect::<Vec<_>>().join("\n")
    );

    // The whole class: an explicitly-chosen family must never report an
    // auto-inference that never happened, in either heuristic direction.
    assert!(
        !combined.contains("Inferred gaussian-identity family"),
        "explicit --family gamma-log wrongly reported 'Inferred gaussian-identity \
         family' — the inference note is gated on link_choice.is_none() instead of \
         the family actually being auto-discovered.\nstderr: {stderr}"
    );
    assert!(
        !combined.contains("Inferred binomial-logit family"),
        "explicit --family gamma-log wrongly reported an inferred binomial-logit \
         family.\nstderr: {stderr}"
    );
}
