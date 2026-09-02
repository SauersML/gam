//! Oracle: the streaming band-assembled K×K tensor-product B-spline smoother
//! must reproduce, to near machine precision, the SAME penalized system built
//! independently in-test — self-constructed truth (#904): naive O(n·p²) dense
//! normal-equation accumulation, naive dense Gauss–Legendre penalty assembly,
//! and in-test Gaussian elimination. Agreement at 1e-8 proves the scatter-add
//! and band assembly are exact (no approximation tolerance budget — both
//! paths compute the same finite-dimensional Gaussian).
//!
//! Three arms:
//! 1. exactness — coefficients, fitted means and prediction variances match
//!    the dense oracle at fixed (λ, σ²);
//! 2. truth recovery — REML-selected fit on a smooth surface + fixed
//!    quasi-random noise beats the noise floor sanely;
//! 3. penalty correctness — for f = x1² + x1·x2 + x2² (constant second
//!    derivatives) the assembled J(f) equals the closed-form integral
//!    (4a1² + 2a1a2 + 4a2²)·Area to 1e-8, which a dropped mixed term
//!    (the axis-wise P-spline shortcut) would miss by exactly 2a1a2·Area.

// ───────────────────────── in-test dense linear algebra ─────────────────────

// ─────────────── in-test re-statement of the basis definition ───────────────
// The basis (uniform extended knots, cardinal cubic segments) is part of the
// model definition shared by both paths; what the oracle re-derives
// INDEPENDENTLY is the assembly (dense accumulation vs streaming band
// scatter-add) and the solve (Gaussian elimination vs Cholesky).

// ───────────────────────────── deterministic data ───────────────────────────

// ──────────────────────────────── arm 1: exactness ──────────────────────────

// ─────────────────────────── arm 2: truth recovery ──────────────────────────

// ───────────────── arm 3: penalty correctness (mixed term) ──────────────────

