//! Finite-difference oracle for the raw B-spline derivative recurrences.
//!
//! The sibling `bspline_derivative_identity_bug.rs` only checks the *first*
//! derivative against the de-Boor recurrence written out by hand — a circular
//! check (recurrence vs. recurrence) that cannot catch a sign/index/order bug
//! in the recurrence itself, and never exercises orders 2/3/4.
//!
//! This test instead pins every analytic derivative order (1..=4) against a
//! central finite-difference stencil of the *value* evaluator
//! (`evaluate_bspline_basis_scalar`), which shares no code with the derivative
//! path. A bug in `evaluate_bspline_derivative_recurrence_into` (the body that
//! powers orders 2/3/4) or in `evaluate_bspline_derivative_scalar` (order 1)
//! shows up as an FD mismatch.
//!
//! The basis is evaluated column-by-column: each column `i` is one scalar
//! function `B_{i,k}(x)`, so we can FD it independently and compare to row `i`
//! of the analytic derivative vector.

