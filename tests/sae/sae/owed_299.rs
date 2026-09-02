//! Issue #299 — arrow_schur preconditioner ladder beyond block-Jacobi.
//!
//! The reduced-Schur PCG used by the bundle-adjustment / SAE-manifold inner
//! step preconditions with a block-Jacobi (per-β-block dense Schur) operator.
//! #299 asks for a richer LADDER — cluster-Jacobi, additive Schwarz,
//! diagonally-assembled Schwarz, and level-0 incomplete Cholesky (IC(0)) — with
//! a measured iteration-count reduction over block-Jacobi on an ill-conditioned
//! arrow system.
//!
//! These gates drive the public study seam
//! `arrow_precond_ladder_iteration_study`, which runs the SAME preconditioned
//! CG (identical rhs, tolerances, trust radius) once per ladder tier and reports
//! each tier's iteration count via `ArrowPcgDiagnostics`. The iteration counts are
//! MEASUREMENTS, not tuned constants; the assertions only encode the structural
//! facts the ladder must satisfy:
//!
//!   1. every tier drives the coupled PCG to convergence (a valid SPD
//!      preconditioner — the correctness check: IC(0)'s `M⁻¹ = (L̃ L̃ᵀ)⁻¹` solve
//!      must let CG reach the requested tolerance);
//!   2. on a system with genuine OFF-block coupling, the richer tiers
//!      (cluster-Jacobi, Schwarz, IC(0)) take strictly fewer iterations than the
//!      block-diagonal preconditioners that cannot see that coupling;
//!   3. on a BLOCK-BANDED coupling (block-tridiagonal reduced Schur) IC(0) — a
//!      no-fill sparse factor that keeps the band — beats block-Jacobi, the
//!      iteration-count reduction #299 measures.

