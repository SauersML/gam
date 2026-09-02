//! Regression for #1765: the Gaussian residual scale `σ̂²` must remain a
//! consistent estimate of the true noise variance even when REML selects a
//! very flexible (high-EDF) fit on low-noise data.
//!
//! The fit-level `standard_deviation` is the observation-noise scale that
//! `predict`'s OBSERVATION interval adds to `Var(μ̂)`:
//!
//!     half_width = z · sqrt( σ̂² + Var(μ̂(x)) ).
//!
//! When the mean is fit very accurately (high EDF, low residuals) the band is
//! dominated by `σ̂²`, so an underestimated `σ̂²` directly collapses
//! observation-interval coverage (the #1765 symptom: coverage < 0.75, PIT KS
//! ~0.18). The unbiased Gaussian scale (mgcv `gam.scale`) is
//!
//!     σ̂² = RSS / (n − edf_total),   edf_total = tr(F) = Σ edf_k,
//!
//! NOT the MLE `RSS / n`, and NOT the null-space `RSS / (n − mp)` divisor
//! (`mp = p − rank(ΣS_k)`): at high EDF the residuals from the flexible fit
//! shrink, and only the full `n − edf` divisor restores `E[σ̂²] = σ²_true`. The
//! #1765 defect was exactly the wrong denominator collapsing σ̂² low; the fix
//! (already in `optimizer.rs`) uses `n − edf_total`. This test locks that in:
//! it FAILS if the denominator regresses to `RSS/n` or `RSS/(n − mp)` and
//! PASSES on the residual-df scale.
//!
//! The fixture uses a full-rank ridge penalty (`S = I`, so `mp = p − p = 0`),
//! which makes both the MLE bug and the null-space bug collapse to the SAME
//! divisor `n`; the assertion below discriminates `RSS/(n − edf)` from `RSS/n`
//! and so catches either regression in one shot.

