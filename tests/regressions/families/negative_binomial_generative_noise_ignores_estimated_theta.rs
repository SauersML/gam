//! Regression for #1124: the generative observation model — `gam generate`,
//! `Model.sample_replicates`, `posterior_predictive_check` — must draw
//! Negative-Binomial responses with the **estimated** overdispersion `theta_hat`,
//! not the construction seed `theta = 1.0`. With the seed, synthetic / replicate
//! counts carry `Var(y) = mu + mu^2` instead of the correct
//! `Var(y) = mu + mu^2/theta_hat` — far too much overdispersion — and the
//! posterior-predictive p-values are wrong.
//!
//! This is the NB sibling of the already-fixed Beta #770 / Tweedie #771 /
//! Gamma #678 generative-dispersion bugs. The shared *root cause* was that the
//! dispersion-picking logic ("given a fit, what scalar dispersion feeds the
//! generative `NoiseModel`?") was **duplicated** across the CLI generate path
//! and the Python `sample_replicates` path — and a third, dead copy — so fixing
//! one copy left the others drawing at the seed. The pickers are now unified
//! into the single `gam::generative::family_noise_parameter`, which both live
//! paths call; this test pins that unified picker on a real NB fit.
//!
//! ## What is asserted (two independent angles)
//!
//!  1. *the fit records the overdispersion*: a real NB fit to data with true
//!     `theta = 3` records `theta_hat` well above the seed `1` in its scale
//!     metadata (`EstimatedNegBinTheta`).
//!  2. *the unified picker → noise composition threads it*: even when the family
//!     spec handed to the picker carries the un-refreshed seed `theta = 1`
//!     (the worst case the generate path can present), the canonical
//!     `family_noise_parameter` recovers `theta_hat` off the fit's scale
//!     metadata, and `NoiseModel::from_likelihood` then carries `theta_hat` for
//!     every row — never the seed.

