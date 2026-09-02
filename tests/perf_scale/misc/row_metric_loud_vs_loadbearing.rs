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
    SaeManifoldAtom::new_with_provided_function_gram(
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
/// (high ThresholdGate logits), so the Euclidean activation fit represents BOTH
/// — nothing is suppressed. The loud atom lives in channel 0 with large
/// amplitude (high presence); the quiet atom in channel 1 with small amplitude
/// (low presence).
fn planted_term() -> SaeManifoldTerm {
    let loud = single_channel_atom("loud_inert", 0, LOUD_AMPLITUDE);
    let quiet = single_channel_atom("quiet_loadbearing", 1, QUIET_AMPLITUDE);

    // Two latent-coordinate blocks (one scalar coord per atom), at t = 0.
    let coord_blocks = vec![Array2::<f64>::zeros((N, 1)), Array2::<f64>::zeros((N, 1))];

    // ThresholdGate logits well above threshold for BOTH atoms on every row, so
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
        AssignmentMode::threshold_gate(0.1, 0.0),
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

