//! End-to-end OBJECTIVE quality: gam's NUTS posterior for a penalized
//! binomial-logit smooth must **recover the known truth and be well
//! calibrated**, not merely reproduce PyMC's draws.
//!
//! The data are generated from a known latent function
//!     η_true(x) = 0.3 + 0.8 · sin(2π x / 10),   y ~ Bernoulli(logit⁻¹(η_true)).
//! Because the generating function is known exactly, the honest quality
//! question is not "does gam's posterior look like PyMC's posterior?" (matching
//! a peer NUTS engine proves nothing — both could be miscalibrated together),
//! but rather:
//!   (A) TRUTH RECOVERY — does the posterior MEAN of the linear predictor
//!       η = Xβ track the true η over the design? We assert
//!       RMSE(gam_post_mean_η, η_true) is below a principled bar set by the
//!       Bernoulli observation noise, and additionally that gam's recovery
//!       error is no worse than PyMC's by more than 10% (match-or-beat on
//!       accuracy — PyMC is the BASELINE, not the target).
//!   (B) CALIBRATION — do gam's pointwise 90% posterior credible intervals for
//!       η actually contain the TRUTH ~90% of the time? We assert empirical
//!       coverage within a tolerance band of the 0.90 nominal level. A correct
//!       Bayesian smoother must be calibrated against the truth; this is an
//!       objective uncertainty claim, independent of any reference tool.
//!
//! PyMC remains in the file as a BASELINE on the same objective metric
//! (truth-recovery RMSE) — gam must match or beat it — and its R-hat is used
//! only to confirm the baseline run itself converged. The pass/fail criteria
//! are gam-vs-truth, never gam-vs-PyMC.
//!
//! A failure here is a real quality shortfall in gam's posterior, never a
//! reason to loosen the bounds or touch gam source.

// Posterior-sampling budget for the REAL prostate arm (n=490, p=47). This arm
// only asserts robust posterior-MEAN held-out probabilities (log-loss / AUC
// match-or-beat) plus an R-hat<1.1 convergence gate — it is NOT a per-point
// credible-interval or sampler-fidelity test, so the draw count need not scale
// with n·p. For an easy log-concave Bernoulli-logit posterior these counts give
// 2×250 = 500 effective draws (MC error on the posterior-mean probability well
// under the 5% match-or-beat tolerance and the 0.02 log-loss margin) and a
// reliably sub-1.1 R-hat, while keeping the silent post-fit sampling block well
// under the 360s suite cap on top of the ~84s GAM REML fit. Tune is matched to
// draws so NUTS step-size / mass-matrix adaptation is fully converged.

