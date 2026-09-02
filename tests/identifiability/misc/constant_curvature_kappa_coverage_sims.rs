//! #944 stage-4 validation sims — the deferred quantitative half of "curvature
//! as an estimand": across REPLICATES of data generated on a known `M_κ`, the
//! profile-likelihood machinery must (1) RECOVER the planted curvature with low
//! bias, (2) COVER the true κ⋆ with its 95% profile CI at ≈ the nominal rate,
//! and (3) hold SIZE on the interior κ=0 flatness test (flat data is not
//! spuriously rejected) while having POWER (curved data is rejected). The
//! single-dataset e2e test (`constant_curvature_kappa_inference_e2e`) asserts
//! sign-recovery and flatness DIRECTION; this test adds the replicate-level
//! calibration the issue charter names ("recovery of κ̂, CI coverage, size of
//! the κ=0 test") — the claims that make "κ̂ = … (95% CI …)" a statistically
//! honest sentence rather than a point estimate.
//!
//! Reference-as-truth: every dataset is generated on a known `ConstantCurvature`
//! geometry and every assertion is against that self-constructed truth or the
//! exact χ² calibration of gam's own profiled REML criterion — never another
//! tool's output. Bars are sized to the small replicate count `R` so they catch
//! a genuinely miscalibrated estimator/CI/test without flaking on binomial noise
//! (kept CI-cheap: small n, few centers, a handful of replicates).

// --- deterministic RNG (splitmix64 → unit / gaussian), no external deps ------

