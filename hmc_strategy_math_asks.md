# Smoothing Parameter Selection Under Extreme Posterior Skewness: Math Asks

## Context for the Math Team

We have a GAM engine that selects smoothing parameters λ = exp(ρ) by
optimizing the Laplace Approximate Marginal Likelihood (LAML / REML). This
works well for most models but fails catastrophically for high-dimensional
radial basis functions with binary outcomes.

### The concrete failing case

- **Response:** binary event (colorectal cancer yes/no)
- **Predictor:** 16-dimensional Duchon spline over genetic PCs
- **Basis:** 30 coefficients (knot centers + polynomial null space)
- **Penalties:** 3 smoothing parameters ρ = (ρ_mass, ρ_tension, ρ_stiffness)
- **Family:** binomial-logit
- **Sample size:** ~22,000

### What goes wrong

1. **REML converges** to some ρ* and returns coefficients β̂.

2. **Skewness diagnostic fires.** We compute per-coefficient posterior skewness:

       s_j = (H⁻¹)_{jj}^{3/2} · T_j

   where H = X'WX + S(ρ) is the penalized Hessian and T_j = Σ_i c_i x_{ij}³
   is the third-derivative projection (c_i = ∂²ℓ/∂η_i² evaluated at the
   working weights). The maximum |s_j| = 345.55, far above our threshold
   of 0.5.

   **Interpretation:** the Laplace approximation to the marginal likelihood
   p(y|ρ) = ∫ p(y|β)p(β|ρ)dβ is unreliable because the integrand is
   extremely non-Gaussian in at least one β direction.

3. **Joint HMC fallback fires.** We attempt to sample (β, ρ) jointly from:

       log p(β, ρ | y) ∝ ℓ(y|β) − ½β'S(ρ)β + ½ log|S(ρ)|₊ + log p(ρ)

   using NUTS with 2 chains, ~2000 warmup + ~2000 samples, diagonal mass
   matrix adaptation, target acceptance 0.8. The sampler uses whitened β
   coordinates: z = L⁻¹(β − μ) where LL' = H⁻¹ at the LAML mode.

4. **HMC fails completely.** ESS = 2, R̂ = 0.000. The sampler is stuck.

5. **Falls back to LAML estimates** — the same estimates we already know are
   unreliable. Then applies an adaptive cubature correction (sigma-point
   integration in the eigenspace of the ρ Hessian), which helps with
   uncertainty quantification but does not fix the point estimates.


## Diagnosis

### Why skewness is extreme

With 16-dimensional Duchon splines and binary outcomes:

- The basis has ~14 knot-derived coefficients + 16 polynomial null-space
  terms (intercept + linear in each PC).
- Many knot coefficients are only weakly identified by the data — their
  posterior variance (H⁻¹)_{jj} can be very large.
- The logit link produces non-zero third derivatives: c_i = μ_i(1−μ_i)(1−2μ_i),
  which are bounded but don't vanish.
- The product (H⁻¹)_{jj}^{3/2} · T_j becomes enormous for weakly-identified
  coefficients, even when T_j itself is moderate.

This is fundamentally a problem of **thin data in high dimensions**: with a
16D input space and ~22K observations, most of the space is sparsely populated,
so many basis functions have little data support.

### Why joint HMC fails

The joint (β, ρ) posterior has **Neal's funnel** geometry:

- ρ controls the scale of β through the penalty S(ρ) = Σ_k exp(ρ_k) R_k'R_k.
- When ρ_k is large (strong penalty), the corresponding β components are
  tightly constrained near zero.
- When ρ_k is small (weak penalty), those β components are free to be large.
- This creates a funnel: narrow in β at large ρ, wide in β at small ρ.

Our whitening z = L⁻¹(β − μ) is computed at a single ρ*, so it only
decorrelates the posterior locally. As the sampler explores different ρ values,
the whitening becomes inappropriate and the effective geometry changes.

Diagonal mass matrix adaptation cannot capture this ρ-dependent scaling.
A dense mass matrix would help but doesn't scale well and still can't capture
the nonlinear ρ-β coupling.

With 33 parameters (30 β + 3 ρ) and funnel geometry, NUTS gets stuck in the
narrow neck of the funnel, producing ESS ≈ 2.


## Math Asks

### Ask 1: Non-centered parameterization

The standard fix for funnel geometry in hierarchical models is the
**non-centered parameterization**:

    β = L(ρ) · z + μ(ρ),    z ~ N(0, I)

where L(ρ)L(ρ)' = S(ρ)⁻¹ (or the conditional prior covariance). Then we
sample (z, ρ) jointly — z has unit-scale prior regardless of ρ, breaking the
funnel.

**Problem:** In our setting, S(ρ) = Σ_k exp(ρ_k) R_k'R_k is a sum of penalty
matrices, and some directions in β-space are unpenalized (the polynomial null
space). So S(ρ) is singular and L(ρ) doesn't exist as a standard Cholesky
factor.

**Questions:**

(a) What is the correct non-centered parameterization when S(ρ) is rank-
deficient? The null space of S(ρ) (the polynomial part of β) is independent
of ρ and should be sampled "centered." Only the penalized part should be
non-centered. Is this a standard split?

(b) The penalty has the form S(ρ) = Σ_k λ_k R_k'R_k where λ_k = exp(ρ_k)
and R_k are fixed matrices. The eigenstructure of S(ρ) changes with ρ because
it's a sum of matrices with different eigenvectors. Is there a practical
decomposition L(ρ) that can be computed efficiently at each HMC step?

(c) The log-posterior in (z, ρ) coordinates is:

    log p(z, ρ | y) = ℓ(y | L(ρ)z + μ(ρ)) − ½z'z + ½ log|S(ρ)|₊ + log p(ρ)
                      + log|det ∂β/∂z| + const

  The Jacobian log|det ∂β/∂z| = log|det L(ρ)|. Is this correct? And does the
  ½ log|S(ρ)|₊ term simplify or cancel with the Jacobian?


### Ask 2: Marginal sampling of ρ only

Instead of sampling (β, ρ) jointly, we could integrate out β and sample ρ
from its marginal posterior:

    p(ρ | y) ∝ p(y | ρ) · p(ρ)

where p(y | ρ) = ∫ p(y|β) p(β|ρ) dβ is exactly the marginal likelihood that
LAML approximates.

The problem is that LAML is a bad approximation here (that's why we triggered
HMC in the first place). But we don't need LAML — we could use a better
approximation to p(y|ρ).

**Questions:**

(a) **Higher-order Laplace:** The Tierney-Kadane correction gives an O(n⁻¹)
improvement. Are there practical O(n⁻²) corrections? Our skewness of 345 seems
to suggest n is effectively very small for the relevant β directions, so
asymptotic corrections may not help. Is this assessment correct?

(b) **Importance-weighted marginal likelihood:** Could we estimate p(y|ρ) by:

    1. Find β̂(ρ) via PIRLS (our inner solver)
    2. Draw β₁, ..., β_M from q(β) = N(β̂, H⁻¹)
    3. Estimate p(y|ρ) = E_q[p(y|β)p(β|ρ)/q(β)]

  Then use this within HMC on ρ only (3 dimensions!)? The gradient would
  need to be estimated too — is there a practical way to get ∂/∂ρ of this
  importance-weighted estimate?

(c) **Bridge sampling or path sampling** for the marginal likelihood — are
these practical for our setting (30-dimensional β integral, need gradients)?

(d) **Nested Laplace (INLA-style):** Compute the marginal p(ρ|y) by
evaluating the Laplace approximation at each ρ, but correct it using the
conditional Laplace approximations p(β_j | ρ, y) individually. Is this
the right framework? Our ρ is only 3-dimensional, so grid or quadrature
over ρ is feasible.


### Ask 3: Diagnosing and fixing the source of skewness

Before trying to sample a difficult posterior, maybe we should ask: **why is
the posterior so skewed, and can we reparameterize to make it less so?**

**Questions:**

(a) The skewness s_j = (H⁻¹)_{jj}^{3/2} · T_j is dominated by the
(H⁻¹)_{jj}^{3/2} factor for weakly-identified coefficients. Would a
**stronger prior** (e.g., increasing the minimum penalty) reduce skewness
without materially changing the fit? Put differently: if a coefficient has
posterior variance 100, it contributes almost nothing to predictions but
dominates skewness. Can we safely regularize these away?

(b) **Basis reparameterization:** Is there a change of basis for the spline
coefficients (e.g., rotating to the eigenvectors of H, or to the penalty
eigenvectors) that would reduce skewness? The skewness depends on both the
posterior shape and the coordinate system — it's not invariant.

(c) **Truncated basis:** With 16 PCs and ~22K observations, some effective
degrees of freedom are nearly zero. Would truncating the basis to only the
well-identified components (e.g., keeping only directions where the effective
degree of freedom > 0.01) reduce skewness while preserving the fit?

(d) We compute skewness as s_j = (H⁻¹)_{jj}^{3/2} · T_j where T_j involves
the third derivative of the log-likelihood. For logit link, the third
derivative is bounded: |c_i| ≤ 1/(6√3). So T_j = Σ_i c_i x_{ij}³ scales
with the design matrix. Is the real problem that our design matrix has
extreme entries (basis functions with very large values at some points)?
Would **rescaling or truncating extreme basis function values** fix this?


### Ask 4: Profile likelihood for ρ

Instead of integrating out β, we could use a **profile approach**: for each ρ,
find β̂(ρ) via PIRLS, then evaluate:

    ℓ_profile(ρ) = ℓ(y | β̂(ρ)) − ½β̂(ρ)'S(ρ)β̂(ρ)

This avoids the marginal likelihood integral entirely. The profile likelihood
is not a true posterior but gives a point estimate for ρ. We already do
something similar (LAML ≈ profile likelihood + log-determinant correction).

**Questions:**

(a) The difference between LAML and the profile likelihood is the
½ log|H(ρ)| − ½ log|S(ρ)|₊ correction (Occam factor). In our failing case,
is it this correction that's unreliable, or is the profile likelihood itself
problematic?

(b) If the Occam factor is the issue: can we compute it more robustly? For
instance, using a spectral decomposition of H and S and computing
log|H|/|S|₊ as a sum of log-ratios of paired eigenvalues (which might be
more numerically stable)?

(c) **Modified profile likelihood (Barndorff-Nielsen):** The modified profile
likelihood adjusts the profile with a term that accounts for the curvature of
β̂(ρ) w.r.t. ρ. Is this applicable here? It's O(1) in n (not asymptotic),
so it might work even when n_eff is small.


### Ask 5: What should the decision framework be?

Currently our fallback chain is:

    LAML → skewness check → HMC (often fails) → keep LAML anyway

This is unsatisfying. We need a decision framework that actually works.

**Questions:**

(a) Given that ρ is low-dimensional (typically 2–6), is there a robust
algorithm that always works for selecting ρ? For instance:

  - **Grid search** over ρ with cross-validated deviance (no Laplace needed)
  - **Grid search** over ρ with importance-sampled marginal likelihood
  - **Bayesian optimization** of the marginal likelihood with a GP surrogate

These are all expensive (each ρ evaluation requires a PIRLS fit) but in low
dimensions they're feasible. What's the right cost/accuracy tradeoff?

(b) How should we detect when LAML is reliable enough to skip these expensive
alternatives? Our current skewness diagnostic catches gross failures, but is
there a tighter criterion? For instance, comparing LAML to a few-point
quadrature estimate and checking agreement?

(c) When HMC does run, what convergence criteria should we use for the
smoothing parameters specifically? R̂ and ESS on ρ (3 parameters) will
converge much faster than on the full (β, ρ) space. Should we monitor ρ
convergence separately and stop early?


### Ask 6: Laplace approximation quality for specific kernel/link combinations

**Questions:**

(a) Is it known that **radial basis function models with logit link** have
particularly poor Laplace approximations? The literature on INLA discusses
Laplace quality for specific model classes — is there guidance for our setting?

(b) The extreme skewness (345) is driven by coefficients with large posterior
variance. In GP/spline regression, these are the "nearly unpenalized"
directions — basis functions in regions with no data. For these directions,
the posterior is essentially the prior (penalty), which IS Gaussian. So why
is the Laplace approximation bad for them?

Our hypothesis: the issue is not the marginal posterior of individual β_j,
but the **conditional** structure — when some β_j have large freedom, their
interaction with the likelihood through the logit link creates skewness in
the joint posterior that the Laplace doesn't capture. Is this correct?

(c) For Gaussian responses (identity link), the Laplace approximation is
exact because the integral is Gaussian. For logit, it's approximate. Is there
a **known bound** on the Laplace error as a function of the link's third
derivative and the posterior spread? Something like:

    |log p(y|ρ) − LAML(ρ)| ≤ C · max_j (H⁻¹)_{jj}^{3/2} · |T_j|

If such a bound exists, we could use it to decide when LAML is trustworthy.


## Summary of Deliverables

1. **Non-centered parameterization** for rank-deficient penalty — correct formulation (Ask 1)
2. **Marginal ρ sampling** — best approach for 3D ρ when Laplace is unreliable (Ask 2)
3. **Source of skewness** — can we reparameterize or regularize it away? (Ask 3)
4. **Profile likelihood** — is the Occam factor the unreliable piece? (Ask 4)
5. **Decision framework** — what algorithm always works for low-dimensional ρ? (Ask 5)
6. **Theoretical bounds** — when is Laplace reliable for kernel + logit? (Ask 6)

## Priority ordering

If only one thing gets answered: **Ask 5a** — we need a method that always
works for 3–6 dimensional ρ, even if it's slower. The current approach of
"try Laplace, if bad try HMC, if HMC fails shrug" is not acceptable for
clinical deployment.

If two things: add **Ask 3a/3c** — if we can reduce skewness by regularizing
or truncating the basis, Laplace might just work and we avoid the whole
problem.
