//! Honest negative control for the amended #980 metric design: the two-score
//! per-atom **lens** ([`gam::inference::atom_lens::atom_two_lens`]).
//!
//! # The corrected headline
//!
//! The *wrong* (original) headline was "the metric the likelihood whitens by
//! decides which structure the reconstruction recovers" — i.e. fold the
//! output-Fisher metric into the SAE loss so the loud-but-inert structure is
//! *suppressed* and only the load-bearing one survives. That is the
//! loss-replacement mistake #980 was amended to remove: it makes the gauge drive
//! the fit and silently deletes anything *represented but not currently used*.
//!
//! The corrected design:
//!
//! * **The SAE fit stays on activations.** The reconstruction loss is Euclidean
//!   (the only loss). *Everything represented survives* — both a loud-but-inert
//!   high-variance structure and a quiet-but-load-bearing low-variance feature
//!   are recovered/represented by the fit. Neither is suppressed.
//! * **The Fisher metric is an additive report, never a loss.** Output-Fisher
//!   factors enter *only* through the lens's `coupling` score; they do not touch
//!   the activation fit. A loud atom that carries (by construction) near-zero
//!   behavioral coupling is **flagged** "represented-not-used" but is **not**
//!   removed; a quiet atom with high coupling is flagged "used".
//!
//! This test plants exactly that situation and asserts the lens reads it
//! correctly. It is the falsifiable negative control for the whole metric design:
//! if the lens ever *suppressed* the loud structure (instead of reporting it), or
//! failed to surface the represented-not-used discrepancy, this test fails.
//!
//! ## Why this is the negative control
//!
//! "Loud-but-inert" is the adversarial case for any metric that drives the loss:
//! a high-variance artifact dominates the residual sum of squares, so a
//! Fisher-weighted *loss* would have to fight it, and a naive whitening would
//! erase it. Here the activation fit keeps it (Euclidean loss, nothing erased),
//! and the lens — reading the synthesized OutputFisher metric — correctly reports
//! that the loud atom's *behavioral* coupling is ~zero while its
//! *representational* presence is large. That gap is the headline safety number.

use std::sync::Arc;

use gam::inference::atom_lens::{AtomTwoLensReport, atom_two_lens};
use gam::inference::row_metric::{MetricProvenance, RowMetric};
use gam::terms::sae::manifold::{
    AssignmentMode, SaeAssignment, SaeAtomBasisKind, SaeManifoldAtom, SaeManifoldTerm,
};
use ndarray::{Array2, Array3};

const N: usize = 200;
const P: usize = 2; // output channels: 0 = loud atom's channel, 1 = quiet atom's channel.

/// Loud-but-inert atom amplitude: a large decoder magnitude living entirely in
/// output channel 0. High representational presence (it is loud), but the
/// synthesized OutputFisher metric carries almost no precision on channel 0, so
/// its behavioral coupling is ~zero.
const LOUD_AMPLITUDE: f64 = 6.0;
/// Quiet-but-load-bearing atom amplitude: a small decoder magnitude living
/// entirely in output channel 1. Low presence, but the OutputFisher metric puts
/// full precision on channel 1, so its behavioral coupling is high.
const QUIET_AMPLITUDE: f64 = 1.0;

/// Build a single-basis atom whose decoder maps the (constant) basis to a single
/// output channel with the given amplitude, and whose basis Jacobian is a unit
/// slope so the decoder tangent `dg/dt` is exactly the decoder row (a nonzero
/// vector living in `channel`). Caller-managed basis (no evaluator): the
/// construction-time `basis_values` / `basis_jacobian` are authoritative, which
/// is all the read-only lens needs.
fn single_channel_atom(name: &str, channel: usize, amplitude: f64) -> SaeManifoldAtom {
    let m = 1usize; // one basis function.
    let latent_dim = 1usize;
    // Constant basis value 1 on every row.
    let mut basis_values = Array2::<f64>::zeros((N, m));
    for row in 0..N {
        basis_values[[row, 0]] = 1.0;
    }
    // Unit slope on every row: dΦ/dt = 1 ⇒ tangent dg/dt = 1 · B_k = B_k.
    let mut basis_jacobian = Array3::<f64>::zeros((N, m, latent_dim));
    for row in 0..N {
        basis_jacobian[[row, 0, 0]] = 1.0;
    }
    // Decoder maps the basis to exactly one output channel with `amplitude`.
    let mut decoder = Array2::<f64>::zeros((m, P));
    decoder[[0, channel]] = amplitude;
    // No roughness penalty: a 1×1 zero Gram (order 0, reweighting skipped).
    let smooth_penalty = Array2::<f64>::zeros((m, m));
    SaeManifoldAtom::new(
        name,
        SaeAtomBasisKind::EuclideanPatch,
        latent_dim,
        basis_values,
        basis_jacobian,
        decoder,
        smooth_penalty,
    )
    .expect("single-channel atom must build")
}

/// Build the planted two-atom fitted term. Both atoms are *active on every row*
/// (high JumpReLU gate logits), so the Euclidean activation fit represents BOTH
/// — nothing is suppressed. The loud atom lives in channel 0 with large
/// amplitude (high presence); the quiet atom in channel 1 with small amplitude
/// (low presence).
fn planted_term() -> SaeManifoldTerm {
    let loud = single_channel_atom("loud_inert", 0, LOUD_AMPLITUDE);
    let quiet = single_channel_atom("quiet_loadbearing", 1, QUIET_AMPLITUDE);

    // Two latent-coordinate blocks (one scalar coord per atom), at t = 0.
    let coord_blocks = vec![Array2::<f64>::zeros((N, 1)), Array2::<f64>::zeros((N, 1))];

    // JumpReLU gate logits well above threshold for BOTH atoms on every row, so
    // both gates are ~1 (σ((10−0)/0.1) ≈ 1). This is the "everything represented
    // survives" condition: the activation fit keeps both atoms maximally active.
    let mut logits = Array2::<f64>::zeros((N, 2));
    for row in 0..N {
        logits[[row, 0]] = 10.0;
        logits[[row, 1]] = 10.0;
    }
    let assignment = SaeAssignment::from_blocks_with_mode(
        logits,
        coord_blocks,
        AssignmentMode::jumprelu(0.1, 0.0),
    )
    .expect("assignment must build");

    SaeManifoldTerm::new(vec![loud, quiet], assignment).expect("term must build")
}

/// Synthesize the OutputFisher factors that make the loud channel near-inert and
/// the quiet channel fully load-bearing *behaviorally*. Per-row rank-1 factor
/// `U_n ∈ ℝ^{P × 1}` with sqrt-precision ~0 on channel 0 (loud) and ~1 on
/// channel 1 (quiet). The Fisher mass of a tangent `x` is `‖U_nᵀ x‖²`:
///   * loud tangent `[LOUD_AMPLITUDE, 0]` ⇒ `(0.02·LOUD_AMPLITUDE)²` ≈ 0,
///   * quiet tangent `[0, QUIET_AMPLITUDE]` ⇒ `(1·QUIET_AMPLITUDE)²`, large.
fn output_fisher_metric() -> RowMetric {
    let rank = 1usize;
    let mut u = Array2::<f64>::zeros((N, P * rank));
    for row in 0..N {
        u[[row, 0]] = 0.02; // loud / channel 0: almost no precision.
        u[[row, rank]] = 1.0; // quiet / channel 1: full precision.
    }
    RowMetric::output_fisher(Arc::new(u), P, rank).expect("OutputFisher metric must be valid PSD")
}

/// Locate an atom's entry by name.
fn entry<'a>(
    report: &'a AtomTwoLensReport,
    name: &str,
) -> &'a gam::inference::atom_lens::AtomLensEntry {
    report
        .atoms
        .iter()
        .find(|e| e.name == name)
        .unwrap_or_else(|| panic!("atom {name} must be in the lens report"))
}

#[test]
fn lens_flags_loud_represented_not_used_and_quiet_used() {
    let term = planted_term();

    // --- The activation fit is Euclidean-on-activations: both atoms present. ---
    // Reading the lens under the Euclidean (default) metric exercises the
    // representational presence axis with NO Fisher. Both atoms must be present
    // (nonzero), because the Euclidean activation fit represents everything — the
    // loud structure is NOT suppressed.
    let euclid = RowMetric::euclidean(N, P).expect("Euclidean metric must build");
    let euclid_report = atom_two_lens(&term, &euclid);
    assert_eq!(
        euclid_report.coupling_provenance,
        Some(MetricProvenance::Euclidean)
    );
    assert!(
        !euclid_report.coupling_available(),
        "Euclidean provenance carries no behavioral coupling axis"
    );
    let loud_e = entry(&euclid_report, "loud_inert");
    let quiet_e = entry(&euclid_report, "quiet_loadbearing");
    assert!(
        loud_e.presence > 0.0 && quiet_e.presence > 0.0,
        "both atoms must be represented by the Euclidean activation fit \
         (loud={}, quiet={})",
        loud_e.presence,
        quiet_e.presence
    );
    assert!(
        loud_e.presence > quiet_e.presence,
        "the loud atom is louder (higher presence) than the quiet one: {} vs {}",
        loud_e.presence,
        quiet_e.presence
    );
    // Graceful degradation: with no Fisher, coupling / discrepancy are None, and
    // nothing is flagged (the lens is optional, the behavioral axis simply absent).
    assert!(loud_e.coupling.is_none() && loud_e.discrepancy.is_none());
    assert!(quiet_e.coupling.is_none() && quiet_e.discrepancy.is_none());
    assert!(!loud_e.is_represented_not_used() && !loud_e.is_used());
    assert!(!quiet_e.is_represented_not_used() && !quiet_e.is_used());

    // --- The additive lens: Fisher enters ONLY here, as a report. ---
    let fisher = output_fisher_metric();
    let report = atom_two_lens(&term, &fisher);
    assert!(matches!(
        report.coupling_provenance,
        Some(MetricProvenance::OutputFisher { .. })
    ));
    assert!(
        report.coupling_available(),
        "OutputFisher provenance makes the behavioral coupling axis available"
    );

    let loud = entry(&report, "loud_inert");
    let quiet = entry(&report, "quiet_loadbearing");

    let loud_coupling = loud
        .coupling
        .expect("loud coupling available under OutputFisher");
    let quiet_coupling = quiet
        .coupling
        .expect("quiet coupling available under OutputFisher");
    let loud_cn = loud
        .coupling_normalized
        .expect("loud normalized coupling available");
    let quiet_cn = quiet
        .coupling_normalized
        .expect("quiet normalized coupling available");

    println!(
        "LOUD : presence={:.6} (norm {:.4})  coupling={:.6} (norm {:.4})  discrepancy={:?}",
        loud.presence, loud.presence_normalized, loud_coupling, loud_cn, loud.discrepancy
    );
    println!(
        "QUIET: presence={:.6} (norm {:.4})  coupling={:.6} (norm {:.4})  discrepancy={:?}",
        quiet.presence, quiet.presence_normalized, quiet_coupling, quiet_cn, quiet.discrepancy
    );

    // PRESENCE is unchanged by the metric (it never touches Fisher): both still
    // present, loud still louder. *Everything represented survives* — the loud
    // atom is NOT suppressed by the presence of a Fisher metric.
    assert!(
        loud.presence > 0.0 && quiet.presence > 0.0,
        "the additive lens must not suppress representation: loud={}, quiet={}",
        loud.presence,
        quiet.presence
    );
    assert!(
        loud.presence > quiet.presence,
        "loud is still the louder (more present) atom under the OutputFisher lens"
    );

    // COUPLING: the synthesized factors put the loud atom's behavioral mass at
    // ~zero and the quiet atom's near its full amplitude. Normalized, quiet is
    // the max-coupling atom (1.0) and loud is near 0.
    assert!(
        loud_cn < 0.05,
        "loud atom must have ~ZERO behavioral coupling (normalized): {loud_cn}"
    );
    assert!(
        quiet_cn > 0.9,
        "quiet atom must carry HIGH behavioral coupling (normalized): {quiet_cn}"
    );
    assert!(
        quiet_coupling > 50.0 * loud_coupling,
        "quiet behavioral coupling must dominate the loud one by a wide margin: \
         quiet={quiet_coupling} vs loud={loud_coupling}"
    );

    // HEADLINE — the loud atom is REPRESENTED-BUT-NOT-USED, and crucially it is
    // *flagged*, not *suppressed*: its presence is still large. "Thinking it, not
    // saying it."
    let loud_disc = loud.discrepancy.expect("loud discrepancy available");
    assert!(
        loud_disc > 0.5,
        "loud atom's discrepancy (presence − coupling) must be a large positive \
         'represented-not-used' signal; got {loud_disc}"
    );
    assert!(
        loud.is_represented_not_used(),
        "loud atom must be FLAGGED represented-not-used"
    );
    assert!(
        !loud.is_used(),
        "loud atom must NOT read as behaviorally used"
    );
    // Not suppressed: it remains the most-present atom in the report.
    assert!(
        (loud.presence_normalized - 1.0).abs() < 1e-12,
        "loud atom must remain maximally present (normalized presence 1.0), i.e. \
         flagged but not removed; got {}",
        loud.presence_normalized
    );

    // The quiet load-bearing atom reads as USED: its behavioral coupling matches
    // or exceeds its presence (non-positive discrepancy).
    let quiet_disc = quiet.discrepancy.expect("quiet discrepancy available");
    assert!(
        quiet_disc <= 0.0,
        "quiet atom's coupling must match-or-exceed its presence (non-positive \
         discrepancy); got {quiet_disc}"
    );
    assert!(
        quiet.is_used(),
        "quiet load-bearing atom must be FLAGGED used"
    );
    assert!(
        !quiet.is_represented_not_used(),
        "quiet atom must NOT read as represented-not-used"
    );

    // Decisive cross-over: SAME activation fit (both represented), opposite
    // behavioral reading — selected solely by the additive Fisher report, never
    // by the loss. This is the honest negative control for the metric design.
    assert!(
        loud.is_represented_not_used() && quiet.is_used(),
        "the lens must separate represented-not-used (loud) from used (quiet)"
    );
}
