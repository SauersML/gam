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

