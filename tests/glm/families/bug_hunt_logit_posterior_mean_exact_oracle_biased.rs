//! Bug hunt (#1459): `quadrature::logit_posterior_mean_exact(mu, sigma)` is
//! documented as the EXACT oracle for `E[sigmoid(eta)]` with `eta ~ N(mu, sigma^2)`,
//! used to certify the cheap Gauss-Hermite `logit_posterior_mean` path. But it
//! carried a systematic, sigma-INDEPENDENT, mu-odd bias toward 0.5: at `mu = 3`
//! it was off by ~3.71e-5 for EVERY sigma, while the GHQ path matched truth to
//! ~1e-8.
//!
//! Root cause: the oracle summed the Faddeeva-series form
//! `1/2 - (sqrt(2*pi)/sigma) * Sum_{n>=1} Im w(z_n)` directly. After the
//! `sqrt(2*pi)/sigma` weighting, the terms decay only as `O(1/n)`, so the fixed
//! truncation (and a magnitude early-exit the slow tail never triggers) left an
//! `O(1/N)` remainder. That remainder is constant in sigma and proportional to
//! mu -- exactly the observed bias. Sharpening the `w(z)` evaluator does NOT fix
//! it; the defect is the truncated slow tail.
//!
//! This test pins the oracle against an INDEPENDENT dense-trapezoid reference
//! for `integral phi(z) * sigmoid(mu + sigma*z) dz` (standard-normal pdf `phi`,
//! NO Gauss-Hermite, NO Faddeeva) on a fine grid. It must FAIL before the fix
//! (errors ~1e-5 at the hard cases) and PASS after (errors < 1e-7).

