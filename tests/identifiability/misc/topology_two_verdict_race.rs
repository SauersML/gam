//! #980 verification §2 — the **circle-read-discretely two-verdict race**,
//! the fixture that demonstrates representational and computational geometry
//! are *different measurable objects* (consuming #907's discrete-mixture rung
//! and cross-class stacking adjudication).
//!
//! # The planted situation
//!
//! A geometric circle is planted in activation space. A synthetic downstream
//! readout consumes it **discretely**: it snaps the circular coordinate to the
//! nearest of `k = 7` arc centers (a sharp von-Mises-weighted average of the
//! arc-center directions — smooth, saturating, exactly the "circle used as 7
//! arcs" of the Engels et al. controversy). Then:
//!
//! * **Representational verdict** — the topology race run on the raw
//!   activation cloud must say **circle** (the shape of `p(x)`).
//! * **Computational verdict** — the race run on the cloud's image under the
//!   readout must say **7-cluster mixture** (the shape of what `F` reads).
//!   Racing the readout image *is* racing under the output-Fisher pullback
//!   geometry: distances in the image are `‖dF‖ = ‖J dx‖`, i.e. `M(x)`
//!   distances with `M = JᵀJ` — realized globally, so the linearization
//!   caveat does not bite.
//!
//! # The amended #980 semantics, asserted
//!
//! The two verdicts are **both reported, neither replaces the other**: "circle
//! in the representation, consumed discretely here" is the *finding*, not a
//! contradiction to resolve by pickinging a metric. The test therefore asserts
//! the two arms disagree — that disagreement is the measurable content of the
//! representational-vs-computational distinction, and it is exactly what
//! eyeballing PCA plots cannot adjudicate.
//!
//! Both arms run the same cross-class machinery: in-class mixture ladder by
//! rank-aware Laplace evidence, then the cross-class race with the **held-out
//! stacking log-density headline** (#907's discipline — Laplace evidence
//! across model classes is corroboration only).

// ---------------------------------------------------------------------------
// Deterministic RNG (fixed integer seed, no clock).
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// The plant: one circle, one quantizing readout.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Smooth-circle (ring) candidate: held-out density provider + rank-aware
// evidence, identical in form to the existing planted races.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// One arm of the race: circle candidate vs the in-class mixture winner, with
// the cross-class stacking headline.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// The two-verdict test.
// ---------------------------------------------------------------------------

