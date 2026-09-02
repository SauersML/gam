//! #1031 consumer-level oracle for `fit_pair_surface`, THE first-class
//! pair-component estimator backed by the streaming 2-D grid engine.
//!
//! Self-constructed truth (#904): an independent dense re-statement of the
//! SAME estimator — naive O(n·p²) normal equations on the uniform cubic
//! B-spline tensor basis, naive dense Gauss–Legendre assembly of the FULL
//! anisotropic biharmonic penalty at the consumer's pinned metric
//! `a_i = span_i²`, in-test Gaussian elimination — must reproduce every
//! carve-facing object the consumer hands out (coefficients, scale-free
//! coefficient covariance, EDF, residual cross-covariance, predictions) to
//! near machine precision, and the REML-selected λ must be a maximizer of
//! the independently computed pooled restricted criterion.
//!
//! Three arms:
//! 1. dense-oracle exactness of the consumer surface (small grid);
//! 2. carve integration — a planted ADDITIVE surface fissions losslessly,
//!    a planted BOUND surface is rejected and stays whole (one empirical
//!    measure end to end: the bases the consumer returns are the bases the
//!    carve centers against);
//! 3. e2e truth recovery at large gridded n (320×320 lattice, two response
//!    dimensions sharing one REML λ) through the streaming path, with
//!    posterior predictions from the consumer's own honest API.

// ───────────────────────── in-test dense linear algebra ─────────────────────

// ─────────────── in-test re-statement of the basis definition ───────────────

// ───────────────────────────── deterministic data ───────────────────────────

// ─────────────────────── arm 1: dense-oracle exactness ──────────────────────

// ───────────────────────── arm 2: carve integration ─────────────────────────

// ──────────────────── arm 3: large gridded n, end to end ────────────────────

