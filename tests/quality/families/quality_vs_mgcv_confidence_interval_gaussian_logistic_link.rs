//! End-to-end quality: gam's confidence-interval construction under a
//! **non-identity (logistic) link** must be *well-calibrated against the known
//! truth* — its nominal-95% intervals must actually cover the true latent
//! function at the nominal rate. `mgcv` is retained only as a **baseline to
//! match-or-beat** on calibration, never as the thing gam must reproduce.
//!
//! OBJECTIVE METRIC (this is the pass/fail claim):
//!   The data are generated from a *known* latent smooth `η(x)`,
//!   `μ(x) = sigmoid(η(x))`, `y ~ Bernoulli(μ)`. Because the truth is known
//!   exactly, we measure the **empirical coverage** of gam's pointwise 95%
//!   confidence intervals across the training grid:
//!     * link scale:     fraction of points with `η(xᵢ) ∈ [eta_lowerᵢ, eta_upperᵢ]`
//!     * response scale: fraction of points with `μ(xᵢ) ∈ [mean_lowerᵢ, mean_upperᵢ]`
//!   pooled over many Bernoulli response replicates on a fixed design.
//!
//! WHY THE DGP MUST BE RECOVERABLE (the load-bearing design choice).
//!   The Nychka/Marra–Wood result for penalized GAMs (Wood 2006 §4.8/§6.10;
//!   Marra & Wood 2012) is that the Bayesian band `Vp = (XᵀWX + ΣλⱼSⱼ)⁻¹·φ`
//!   attains ~nominal **across-the-function** coverage of the truth. That
//!   guarantee holds in the regime where the penalized estimator's squared
//!   *bias* is comparable to (not dominated by) its variance — i.e. when the
//!   data actually inform the smooth well enough that REML does not collapse it
//!   toward a near-null fit. The Bayesian covariance encodes the prior-implied
//!   bias-variance trade-off; it CANNOT encode bias that the smoothing
//!   parameter has effectively defined away. If the truth is too wiggly to be
//!   resolved at the given sample size, REML *correctly* over-smooths, the fit
//!   carries a large `O(λ·f'')` bias at every crest/trough, and the band — gam's
//!   OR mgcv's — under-covers the truth no matter how well the variance is
//!   propagated. Pooling Bernoulli **response** replicates at a fixed,
//!   under-informed design does not rescue this: the smoothing bias is
//!   systematic across replicates (it is a property of the design and the
//!   REML-selected λ, not of the response noise), so the replicate-pooled
//!   average estimates a coverage that is genuinely below nominal — it is the
//!   coverage of a bias-dominated band, not the Nychka object. (Empirically, on
//!   a 6-cycle saturating logit DGP at n=200 BOTH gam and mgcv pool to
//!   ~0.45–0.68; only when n grows enough for REML to resolve the signal — EDF
//!   rising from ~3 to ~20 around n≈2000 — does mgcv's pooled coverage snap back
//!   to ~0.95. The band machinery was correct the whole time; the n=200 design
//!   simply did not carry the information.)
//!
//!   We therefore generate from a smooth that IS recoverable at the chosen n:
//!   `η(x) = 2·(x − ½) + 2·sin(3πx)` on `x ∈ [0, 1]` (a gentle slope plus a
//!   1½-cycle sinusoid), `n = 300`. The latent stays away from the saturated
//!   tails (`μ ∈ ≈[0.12, 0.94]`), so the Binomial Fisher information `μ(1−μ)`
//!   never collapses and `k = 15` puts the truth comfortably inside the basis
//!   span. In this regime REML resolves the signal (EDF ≈ 8–9, well above the
//!   over-smoothed ~3 floor and below k), bias ≲ variance, and the across-the-
//!   function coverage claim is well-posed: both engines land at the nominal
//!   level. This is the logit analogue of the identity-link sibling sweep test,
//!   not a weakened bound — a genuinely mis-scaled band still fails here.
//!
//! Why a Binomial(logit) model: this is the family that actually exercises gam's
//! inverse-link Jacobian `dμ/dη = μ(1−μ)` inside CI construction (the Gaussian
//! posterior-variance branch ignores the link entirely). The fixed design is
//! drawn once (seed=123); the Bernoulli responses are then redrawn for each
//! replicate from the same true `μ(x)` so coverage is measured over the
//! response sampling distribution at a fixed configuration of `x`.
//!
//! Identical data feed both engines (the same CSV columns). Bounds are not
//! weakened to force a pass: a genuinely mis-calibrated band failing here is a
//! real bug.

