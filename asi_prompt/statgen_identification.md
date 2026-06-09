# Statistical-genetics identification and survival-warp gauge problems

This document poses four self-contained mathematical problems in semiparametric
identification, gauge theory of penalized likelihoods, and optimal estimation.
No software, dataset, or computational resource is referenced or required. Every
object is defined from first principles. Each problem asks for the most general
provably-correct derivation, proof, identification result, or optimal estimator.

---

## 0. Notation and common objects

Throughout, capital letters denote random variables and lower-case letters their
realizations. For a random vector \(V\), \(\mathbb{E}[\cdot\mid V]\) is the
conditional expectation given \(V\), and \(L^2(P)\) is the Hilbert space of
square-integrable functions of the relevant random elements under the data law
\(P\), with inner product \(\langle f,g\rangle = \mathbb{E}[f g]\).

- \(C \in \mathbb{R}^{d}\): an observed covariate vector. In the genetics
  problems \(C\) is a vector of **principal components (PCs)** summarizing
  genetic ancestry; we write \(C\) and "PC" interchangeably.
- \(Z \in \mathbb{R}\): a scalar **score** (a polygenic score, PGS). Its
  conditional law given \(C\) is described by a **conditional mean**
  \(m(C) := \mathbb{E}[Z\mid C]\) and a **conditional variance**
  \(s^2(C) := \mathrm{Var}(Z\mid C)\); write \(Z = m(C) + s(C)\,R\) with
  \(\mathbb{E}[R\mid C]=0\), \(\mathrm{Var}(R\mid C)=1\).
- \(Y\): an outcome (binary \(Y\in\{0,1\}\) or real-valued).
- \(T>0\): an event time (survival problem).
- \(\phi,\Phi\): the standard normal density and CDF; \(\Phi^{-1}\) its quantile
  function.
- A **monotone link** \(g:\mathbb{R}\to\mathbb{R}\) with mean response
  \(\mu = g^{-1}(\eta)\) and linear predictor \(\eta\).
- A **penalty** is a symmetric positive-semidefinite quadratic form
  \(\tfrac12\,\beta^\top S \beta\) on a finite-dimensional coefficient vector
  \(\beta\); \(\lambda>0\) is a smoothing parameter and \(\rho=\log\lambda\). The
  null space \(\ker S\) is the unpenalized subspace.
- A **basis expansion** of a function \(f(\cdot)\) is
  \(f(c) = \sum_j \beta_j B_j(c) = B(c)^\top\beta\) for fixed basis functions
  \(B_j\); a **smooth** \(s(\cdot)\) is such an expansion equipped with a penalty.
- **Rank inverse-normal transform (rank-INT)** of an i.i.d. sample
  \(z_1,\dots,z_n\): \(\tilde z_i = \Phi^{-1}\!\big(\tfrac{\mathrm{rank}(z_i)-c}{n-2c+1}\big)\)
  for an offset \(c\in[0,\tfrac12]\) (e.g. Blom \(c=\tfrac38\)); as
  \(n\to\infty\) this is the population map \(z\mapsto \Phi^{-1}(F_Z(z))\) where
  \(F_Z\) is the **marginal** CDF of \(Z\).

---

## Problem 1 (headline): Conditional-shift identification of a marginal-slope model

### 1.1 Setup, from first principles

We observe i.i.d. tuples \((Y_i, Z_i, C_i)_{i=1}^n\). The scientific target is a
**marginal slope** \(b(C)\): the local effect of the score \(Z\) on the outcome
\(Y\), allowed to vary with ancestry \(C\). The working model is a generalized
additive model

\[
\eta_i \;=\; b(C_i)\,Z_i \;+\; q(C_i), \qquad
\mathbb{E}[Y_i\mid Z_i,C_i] \;=\; g^{-1}(\eta_i),
\]

where:

- \(b(\cdot)\) is the **slope surface** of interest (a smooth of \(C\)), and
- \(q(\cdot)\) is an **influence / score channel**: a smooth of \(C\) that
  absorbs the ancestry-direct effect on \(Y\). It plays the role of a control
  function; it is *not* the estimand but it shares the design space with \(b\).

The slope \(b\) and the channel \(q\) are estimated jointly (e.g. by penalized
likelihood). The conditional law of \(Z\) given \(C\) carries a **conditional
mean shift** \(m(C)=\mathbb{E}[Z\mid C]\) — in genetics this is the
allele-frequency / ancestry-driven mean of the PGS, generically nonzero — and a
**conditional variance** \(s^2(C)=\mathrm{Var}(Z\mid C)\), generically
non-constant (heteroskedastic).

A common automatic preprocessing **gate** inspects only the **marginal** law of
\(Z\): it computes the empirical marginal moments (skewness, kurtosis), runs a
Kolmogorov–Smirnov test of \(F_Z\) against a reference, and, if non-Gaussian,
applies the population rank-INT map \(z\mapsto\Phi^{-1}(F_Z(z))\) so that the
transformed score \(\tilde Z\) is **marginally** \(N(0,1)\). Crucially, this gate
is a functional of \(F_Z\) alone; it sees nothing of the conditional law
\(Z\mid C\).

### 1.2 What to prove

**(a) Exact conditional-shift bias.** Decompose \(Z = m(C) + s(C) R\) as above.
Treating \((b,q)\) as smooth functions in an \(L^2\) function class and writing
the population estimating equations of the joint penalized fit, derive in closed
form the **asymptotic bias** in the estimated slope functional, i.e. the
difference between the probability limit \(\hat b(\cdot)\) of the joint fit and
the true \(b(\cdot)\), as a functional of \(m(\cdot)\), \(s(\cdot)\), the link
\(g\), and the design (the basis/penalty pair shared by \(b\) and \(q\)). Show
explicitly how the product term \(b(C)\,m(C)\) **leaks into the channel** \(q\)
(i.e. is aliased onto \(q\)'s design), and how a non-constant \(s(C)\) induces an
additional heteroskedastic-weighting bias even when \(m\equiv0\). State the
exact null condition (a condition on \((m,s)\) and the design) under which the
joint plug-in estimator of \(b\) is unbiased.

**(b) Invariance / non-removal under marginal rank-INT.** Prove the central
claim: the rank-INT map \(Z\mapsto\Phi^{-1}(F_Z(Z))\) (and, more generally,
**any** measurable transform \(T(Z)\) that is a function of \(Z\) alone)
**cannot** remove a conditional shift in the relevant sense. Precisely:
characterize the orbit of conditional laws \(\{Z\mid C\}\) reachable by
marginal-only transforms \(T(Z)\), and prove that the conditional-centering
defect \(\mathbb{E}[T(Z)\mid C]\not\equiv \text{const}\) is invariant on this
orbit whenever \(m(\cdot)\) is non-degenerate. Give the exact special case in
which a marginal transform *does* fix the bias (identify the precise
joint-law condition — e.g. a location-family / sufficiency condition relating
\(F_{Z\mid C}\) to \(F_Z\)).

**(c) Optimal minimal conditional diagnostic.** Construct the
**optimal (most powerful, minimal) statistic** \(D_n=D_n(Z_{1:n},C_{1:n})\) that
tests the null
\[
H_0:\quad \mathbb{E}[Z\mid C]\equiv\text{const}\ \text{ and }\ \mathrm{Var}(Z\mid C)\equiv\text{const}
\]
(no conditional mean shift and homoskedastic) against the alternative that a
conditional re-centering / conditional transform is required for valid
marginal-slope inference. Derive its **exact null distribution** (or its exact
asymptotic null law with the precise degrees of freedom / covariance), prove
local optimality (e.g. as a score/Rao test against the worst-case local
alternative in the direction that biases \(b\)), and prove it is **minimal** in
the sense that no statistic of strictly smaller dimension attains the same
local power against the bias-relevant alternatives.

**(d) Most general identifying correction; uniqueness.** Characterize the
**most general transform of \(Z\) conditional on \(C\)** — i.e. a map
\((z,c)\mapsto \zeta = \Psi(z,c)\) — that restores valid marginal-slope inference
(makes the joint plug-in estimator of \(b(\cdot)\) consistent for the true slope
for every smooth \(b\)). Prove that within the class of corrections that (i)
preserve the slope estimand and (ii) are minimal (do not remove any
identifiable signal about \(b\)), there is a **unique minimal sufficient
correction**, and exhibit it (the natural candidate is the conditional
location–scale standardization \(\zeta = (z-m(c))/s(c)\); prove it is the unique
minimal sufficient correction, or characterize the exact equivalence class if
uniqueness holds only up to an affine-in-\(c\) gauge that the channel \(q\)
absorbs).

### 1.3 Neyman-orthogonal estimating equation (the debiasing target)

Let the nuisance be \(\eta = (m(\cdot),s(\cdot))\) (the conditional calibration
of \(Z\)) together with the channel \(q\). Write a plug-in score for \(b\) of the
form \(\psi_i(b,\eta) = a(C_i)\,R_i\,\{Y_i-\mu_i(b,\eta)\}\) with \(R_i=(Z_i-m(C_i))/s(C_i)\)
and a user weight \(a(\cdot)\).

Derive the **Neyman-orthogonalized score** \(\bar\psi\) whose Gateaux
(pathwise) derivative with respect to the nuisance \(\eta\) vanishes at the
truth:
\[
\left.\frac{\partial}{\partial t}\,\mathbb{E}\big[\bar\psi(b_0,\eta_0+t\,h)\big]\right|_{t=0} = 0
\quad\text{for all admissible perturbations } h.
\]
Explicitly compute the nuisance Gateaux derivative of the plug-in score
(both the \(\partial/\partial m\) and \(\partial/\partial s\) directions), give
the orthogonal projection \(\Pi_\eta\) onto the nuisance tangent space that
removes it, and write the resulting orthogonalized estimating equation
\(\bar\psi = \psi - \Pi_\eta[\psi]\). Prove the resulting estimator of \(b\) is
**first-order insensitive** to misestimation of \((m,s)\) (a second-order
remainder bound in the product of nuisance errors), and identify the
**efficient** weight \(a(\cdot)\) that minimizes the asymptotic variance of
\(\hat b(\cdot)\) within the orthogonal class (the semiparametric efficiency
bound for the slope functional). Connect (d) and the orthogonal score: show that
the unique minimal sufficient correction of (d) is precisely what makes the
naive plug-in score coincide with its own orthogonalization to first order.

---

## Problem 2: Survival transformation-AFT time-warp gauge (solved exemplar + open generalization)

### 2.1 Setup, from first principles

Consider an **accelerated failure-time (AFT) transformation model** for a
positive event time \(T\) given covariates \(x\). A **monotone time transform
(warp)** \(h:(0,\infty)\to\mathbb{R}\), strictly increasing with derivative
\(h'>0\), maps the time axis to a real residual axis. A **location–scale
residual** model is imposed:
\[
u \;=\; \frac{h(T) - \mu(x)}{\sigma}, \qquad u \sim \text{standard density } f_0,
\]
with location \(\mu(x) = \beta^\top x\) (the AFT "slopes"), scale \(\sigma>0\),
and a fixed standard density \(f_0\) (take \(f_0=\phi\), standard normal, giving
the log-normal-type AFT; the argument is general). For an observed event at time
\(t\) the log-density of \(T\) is, by change of variables,
\[
\ell(t\mid x) \;=\; \log f_0\!\big(u(t)\big) \;-\; \log\sigma \;+\; \log h'(t),
\qquad u(t) = \frac{h(t)-\mu(x)}{\sigma}.
\]

**Reduced (parametric) regime.** Suppose the warp collapses to its affine null
space in the coordinate \(v(t)=\log t\):
\[
h(t) \;=\; a + b\,v(t) \;=\; a + b\log t,
\]
so \(h'(t) = b/t\) and \(\log h'(t) = \log b - \log t\). The free parameters are
the warp intercept \(a\), the **warp slope** \(b\), the location slopes
\(\beta\), and the scale \(\sigma\).

### 2.2 What to prove

**(a) The exact gauge group, and the Jacobian term that breaks it.** Consider
the reparameterization, for \(c>0\),
\[
(b,\sigma,\beta) \;\longmapsto\; (c\,b,\; c\,\sigma,\; c\,\beta),\qquad a\mapsto c\,a.
\]
Prove that the standardized residual \(u(t)\) is invariant under this map. Show
that the model is unidentified along this **1-parameter multiplicative gauge**
\(c\) **iff** the log-density is invariant. Then prove that the event Jacobian /
normalizer contribution
\[
-\log\sigma + \log h'(t) \;=\; -\log\sigma + \log b - \log t
\]
is, for the warp \(h=a+b\log t\), **invariant** under the gauge (the
\(-\log\sigma\) and \(+\log b\) terms cancel the gauge scaling), so a flat ridge
exists; and conversely prove that pinning \(b\equiv1\) (the convention
\(h(t)=\log t\), as in standard parametric AFT) **removes** the ridge: with
\(b\) fixed, the surviving normalizer \(-\log\sigma\) is **not** gauge-invariant,
which is exactly the term that identifies \(\sigma\) (and hence the absolute
scale of \(\beta\) and of the survival surface \(S(t\mid x)\)). State the result
as: the degeneracy group is \(\mathbb{R}_{>0}\) acting by the scaling above, and
the \(-\log\sigma\) Jacobian term is the unique obstruction generator that
breaks it once \(b\) is fixed.

**(b) Flexible regime: free monotone I-spline warp.** Now let \(h\) be a free
**monotone spline** built from an integrated-spline (I-spline) basis:
\(h(t) = a + \sum_{k} w_k\, M_k(v(t))\) with weights \(w_k\ge0\) (monotonicity)
and \(M_k\) the I-spline basis on \(v=\log t\). The affine null space of the warp
is \(\mathrm{span}\{1, v\}\) (constant + linear-in-\(\log t\)). Characterize the
**residual gauge degeneracy** of \((h,\mu,\sigma)\): show precisely which
directions in the joint parameter space (the affine null-space component of \(h\)
— its overall additive level and its overall multiplicative slope — versus
\(\mu\) and \(\sigma\)) are unidentified, i.e. the largest subgroup of
transformations leaving \((u(t), \ell)\) invariant. Then derive the **minimal
identifying constraint** on the spline's affine null space that (i) pins the
multiplicative scale (fixes \(\sigma\)) and the additive level, yet (ii) **does
not bias the fitted survival function** \(S(t\mid x)\) — i.e. the constraint must
lie entirely in the gauge (non-identified) directions and leave every
identified, survival-relevant degree of freedom free. Prove minimality and
prove the no-bias property (the constrained MLE has the same fitted
\(S(\cdot\mid\cdot)\) as the ridge-restricted true model).

**(c) General principle.** State and prove a **general identification principle
for transformation models with a learned monotone warp**: in a model
\(u = (h(T)-\mu(x))/\sigma\) with \(h\) monotone and partly free, characterize
exactly when the scale \(\sigma\) (equivalently the overall multiplicative gauge
of the warp) is identified, in terms of (i) the contribution of the event
Jacobian \(\log h'\) and (ii) the parameterization of the affine null space of
\(h\). Give the canonical-gauge-fixing recipe (which functional of \(h\) must be
pinned, and to what value) that holds for an arbitrary standard density \(f_0\)
and an arbitrary monotone warp family, and prove it yields a model that is both
identified and statistically equivalent (same likelihood-maximizing fitted
survival) to the unconstrained ridge.

---

## Problem 3: Multinomial softmax penalized REML — canonical gauge and the seed-screening collapse

### 3.1 Setup, from first principles

Let \(Y\in\{1,\dots,K\}\) be a categorical response with \(K\) classes. The
**softmax (multinomial-logit) GAM** models, for class \(c\),
\[
P(Y=c\mid x) \;=\; \frac{\exp(\eta_c(x))}{\sum_{k=1}^{K}\exp(\eta_k(x))},
\qquad
\eta_c(x) \;=\; \sum_{t} f_{c,t}(x),
\]
where each \(f_{c,t}\) is a penalized **smooth** (basis expansion + penalty
\(\tfrac12\lambda_{c,t}\,\beta_{c,t}^\top S_t\,\beta_{c,t}\)) or an unpenalized
linear term. The softmax has an exact **sum-to-zero gauge**: adding any
function \(\delta(x)\) to all \(\eta_c\) leaves the probabilities unchanged, so
only \(K-1\) of the \(K\) class-predictors are identified (conventionally a
reference class \(\eta_K\equiv0\)).

The model is fit by **REML / Laplace-approximate marginal likelihood**: with
\(\rho_{c,t}=\log\lambda_{c,t}\) the vector of log-smoothing parameters, the REML
criterion is
\[
V(\rho) \;=\; -\tfrac12\log\det\!\big(H(\rho)+S(\rho)\big)_{+}
\;+\;\tfrac12\log\det S(\rho)_{+}
\;-\; \ell_p(\hat\beta_\rho)
\;+\;\tfrac12\hat\beta_\rho^\top S(\rho)\hat\beta_\rho,
\]
(generalized determinants over the appropriate ranges), where \(H\) is the
penalized **Fisher information / working Hessian** of the multinomial
log-likelihood at the inner optimum \(\hat\beta_\rho\), and the per-observation
multinomial working weight is the \((K-1)\times(K-1)\) matrix
\[
W \;=\; \mathrm{diag}(p) - p\,p^\top, \qquad p=(p_1,\dots,p_{K-1}).
\]
Near a **saturated / separated** fit, \(p\) approaches a vertex of the simplex
and \(W\to 0\); the Hessian becomes ill-conditioned. A practical fit screens a
set of candidate **starting seeds** \(\rho^{(0)}\) and accepts one whose inner
Newton problem is well-posed.

### 3.2 What to prove

**(a) Why the gauge null direction collapses seed screening.** Characterize the
**canonical gauge** of the penalized multinomial REML problem: the group of
parameter transformations leaving \(V(\rho)\) and the fitted probabilities
invariant (the sum-to-zero / reference-class action plus any aliasing among
replicated cross-class smooths and a smooth's polynomial null space). Prove that
when two coefficient blocks share the same gauge priority (are mutually aliased
under the canonicalizer), the joint penalized Hessian \(H+S(\rho)\) retains a
**structural null direction** \(v\neq0\) with \((H+S(\rho))v=0\) for every
\(\rho\) (the penalty does not touch the gauge direction). Prove that this null
direction makes the generalized-determinant REML terms and the inner Newton
acceptance test degenerate **for all seeds** (the inner system is singular along
\(v\) regardless of \(\rho\)), explaining the universal seed rejection. Give the
exact rank condition on \((H,S(\rho))\) under which the collapse occurs.

**(b) Gauge-invariant REML criterion and seed construction.** Derive a
**gauge-invariant** form of \(V(\rho)\): replace the raw determinants by
generalized determinants over the **identified** quotient space
(coefficients modulo the gauge group), equivalently project \(H+S(\rho)\) onto
the orthogonal complement of the gauge null space and take the determinant of
the projected operator. Prove this quotient REML criterion is (i) well-defined
and finite under the sum-to-zero null space and the \(W\to0\) degeneracy, (ii)
invariant under the canonical gauge group, and (iii) agrees with the standard
REML criterion whenever the gauge is fully fixed. Construct a **well-posed seed**
\(\rho^{(0)}\) (and an inner-Newton acceptance test) that is guaranteed to pass
in the quotient space, i.e. that never rejects solely because of a gauge null
direction.

**(c) Optimal preconditioner near separation.** Derive the **optimal
preconditioner** \(M\) for the penalized softmax Hessian \(H+S(\rho)\) under
near-separation (\(W\to0\)): characterize the minimizer of the spectral
condition number of \(M^{-1/2}(H+S(\rho))M^{-1/2}\) restricted to the identified
quotient space, in terms of the simplex geometry of \(p\) (the structure of
\(\mathrm{diag}(p)-pp^\top\)) and the penalty null space \(\ker S_t\). State it
in closed form (e.g. the block-Jacobi / ridge-stabilized form on the quotient)
and prove its optimality (or near-optimality with an explicit constant) within
the class of symmetric positive-definite preconditioners that respect the gauge
quotient.

### 3.3 Brief addendum (#903): fixed-df vs REML/empirical-Bayes BLUP for random slopes

Consider a Gaussian linear mixed model with a **random slope**: for subject
\(j\), \(y_{ij} = (\beta_0 + b_{0j}) + (\beta_1+b_{1j})\,t_{ij} + \varepsilon_{ij}\),
with \((b_{0j},b_{1j})\sim N(0,G)\), \(\varepsilon\sim N(0,\sigma^2 I)\). A
penalized-smooth / GAM representation rewrites the random effects as a ridge
penalty with covariance \(G\) playing the role of \(\lambda^{-1}\). Derive
exactly the **empirical-Bayes BLUP** predictor for a held-out subject's slope and
intercept, the implied **shrinkage** of \((b_{0j},b_{1j})\), and the **effective
degrees of freedom (EDF)** allocated to the random-slope block as functions of
\((G,\sigma^2,\) per-subject design\()\). Prove the exact correspondence between
the mixed-model REML BLUP and the penalized-GAM predictor, and characterize the
precise condition (on how \(G\), \(\sigma^2\), and the EDF/shrinkage are
estimated and allocated across the intercept vs slope variance components) under
which the GAM random-slope predictor's held-out forecast **coincides** with the
mixed-model BLUP, versus the exact discrepancy when the EDF allocation differs
(e.g. when a single fused smoothing parameter is used in place of a full
\(2\times2\) covariance \(G\)).

---

## Problem 4 (synthesis): A unified gauge-and-orthogonality identification theorem

All three problems above are instances of a single phenomenon: a penalized /
transformation likelihood possesses a **gauge group** \(\mathcal{G}\) (a Lie
group acting on the parameter space leaving the likelihood — or the
likelihood-plus-Jacobian — invariant), and valid inference for a target
functional requires (i) fixing the gauge by a constraint lying entirely in the
non-identified directions, and (ii) orthogonalizing the target's estimating
equation against the remaining nuisance tangent space.

State and prove a **general theorem** unifying the three: given a smooth
parametric-or-semiparametric model with a gauge group \(\mathcal{G}\) acting on
parameters, a target functional \(\theta\mapsto b(\theta)\), and a nuisance
tangent space \(\mathcal{T}_\eta\), characterize (a) the **unique minimal
gauge-fixing constraint** that identifies the scale/level directions without
biasing any identified functional (covering both the conditional
location–scale correction of Problem 1(d) and the warp-slope pin of
Problem 2(b)); (b) the **Neyman-orthogonal score** for \(b\) on the quotient
\(\Theta/\mathcal{G}\), with its semiparametric efficiency bound; and (c) the
**optimal preconditioner / gauge-invariant criterion** on the quotient
(covering Problem 3). Give precise regularity conditions, prove existence and
uniqueness, and exhibit how each of Problems 1–3 is recovered as a special
case with its specific gauge group and nuisance tangent space.
