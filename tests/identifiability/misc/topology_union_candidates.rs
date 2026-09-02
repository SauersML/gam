//! #907 — structured-union composites in the topology race.
//!
//! A *union* candidate is a small FIXED composite of named component structures
//! ({circle+circle, circle+point-cluster, line+cluster}) joined by a hard
//! row-responsibility split. Each component is fit on its group, then the union
//! is scored as one normalized soft-mixture density on every row. BIC prices all
//! component parameters plus the free mixing weight on the common sample scale,
//! so a union can only win when the structured split buys enough likelihood to
//! pay for its extra parameters.
//!
//! These planted tests sample from GROUND-TRUTH generators at fixed integer seeds
//! (no clock randomness) and assert the cross-class adjudicator recovers the
//! planted truth:
//!
//!   * two well-separated circles → a structured union BEATS the single-torus and
//!     the single-circle pure rungs;
//!   * circle + outlier cluster   → a structured union BEATS both pure rungs;
//!   * NEGATIVE CONTROL: a single circle must NOT prefer any union — the
//!     complexity pricing earns its keep and a pure rung carries the headline.
//!
//! The assertions are against the PLANTED TRUTH (which generator produced the
//! data), never against a reference tool's output.

// ---------------------------------------------------------------------------
// Deterministic RNG (fixed integer seed, no clock).
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Planted generators.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Smooth pure-rung held-out density providers (genuine smooth-class candidates).
//
// Single-circle (ring): the data lives on ONE ring with a learned radius
// mean/variance about the data centroid and a uniform-in-angle distribution.
//   p(x, y) = N(r; r_bar, sigma_r^2) * (1 / (2 pi)) * (1 / r)   (polar Jacobian).
//
// Single-torus: a 2-D anisotropic Gaussian (a flat "torus patch" treated as a
// single smooth chart) — the natural smooth competitor that a union of two
// circles or a circle+cluster must beat. Refits per fold.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Race driver: a structured union competes cross-class against the smooth rungs.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Tests.
// ---------------------------------------------------------------------------

