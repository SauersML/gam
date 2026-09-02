//! Object 4 — the Certificate (`residual_gauge`) end-to-end tests.
//!
//! These tests assert the certificate names EXACTLY the gauge group a fitted
//! SAE-manifold model is identified up to. Replicate fits (planted, fixed seeds)
//! must agree up to exactly that named group: both
//!
//!   * under-claiming — reporting a residual freedom the data/isometry pin
//!     actually removes, and
//!   * over-claiming — omitting a real residual freedom
//!
//! must FAIL the test. Two fixtures are constructed by hand so the truth is
//! analytic: one with a genuine residual rotation freedom (isometry pin
//! inactive) and one where the isometry pin removes that same rotation.

