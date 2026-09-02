//! Regression: a prior weight of exactly `0` must make an observation
//! equivalent to its absence (R's `glm` uses `n.ok = nobs − Σ[w==0]`; mgcv
//! drops zero-weight observations). A zero-weighted row contributes exactly
//! zero to every weighted cross-product (`XᵀWX`, `XᵀWy`) and to the weighted
//! RSS (`w_i·r_i² = 0`), so the ONLY channel by which it could still perturb
//! the fit is an explicit observation count `n`. If `n` counts zero-weight
//! rows, the dispersion denominator `weighted_rss / (n − edf)` puts a numerator
//! that already excludes them over a denominator that doesn't, biasing φ̂ low,
//! shrinking every SE, and (via the REML `n` term) shifting the selected λ.
//!
//! This test isolates the count from every other moving part:
//!   - Reference fit B: the deterministic base dataset (n=200), no weights.
//!   - Augmented fit A: the SAME base dataset with a byte-identical second copy
//!     of every row stacked underneath, the copies weighted 0 and the
//!     originals weighted 1.
//! A's positive-weight observations are exactly B, so a correct implementation
//! makes A and B identical. The test asserts λ, EDF, dispersion φ̂, and the
//! fitted coefficients all match to machine precision (#584).

