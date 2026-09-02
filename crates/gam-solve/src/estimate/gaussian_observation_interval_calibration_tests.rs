//! Regression for #1765: the Gaussian OBSERVATION prediction interval must be
//! calibrated in the low-noise / high-EDF regime (the issue reported coverage
//! < 0.75 and PIT KS ~0.18).
//!
//! The observation band a new response `y* = μ(x*) + ε` is covered by is
//!
//!     μ̂(x*) ± z · sqrt( Var(μ̂(x*)) + σ̂² ),
//!
//! i.e. the fitted-mean POSTERIOR variance `Var(μ̂) = x*ᵀ Vp x*` PLUS the
//! observation-noise variance `σ̂²`. Two independent mistakes each collapse this
//! band and produce the #1765 undercoverage, and this test pins both:
//!
//!   1. A residual scale `σ̂²` biased low. The correct Gaussian scale is the
//!      mgcv `gam.scale` residual-df estimate `σ̂² = RSS/(n − edf_total)`, NOT
//!      the MLE `RSS/n` nor the null-space `RSS/(n − mp)` divisor. At high EDF
//!      the flexible fit shrinks the residuals, so the MLE/null-space divisor
//!      lands σ̂ well below σ_true and the band is too narrow.
//!
//!   2. Dropping the mean-posterior term `Var(μ̂)`. At LOW edf/n it is
//!      negligible, but at HIGH edf/n (this fixture: edf/n ≈ 0.28) it is a large
//!      fraction of the total predictive variance — here ≈ 28% of σ² — so an
//!      observation band that adds only `σ̂²` undercovers materially.
//!
//! The production observation band (`gam_predict::family_observation_band`,
//! Gaussian arm) is `sqrt(etavar + σ̂²)`, using the smoothing-corrected `Vp`
//! covariance for `etavar` and the residual-df `σ̂`. This test reconstructs that
//! exact band on a real REML fit and asserts nominal held-out coverage and a
//! small PIT KS, and separately demonstrates that both ingredients are
//! load-bearing (the MLE scale and the mean-term-dropped band both undercover).

