# Open Mathematical Problems in a Penalized-Likelihood / GAM Engine — A Brief for a Pure-Reasoning Solver

## Who you are and what you can (and cannot) do

You are a problem-solver of extreme mathematical and conceptual ability. You have **no access to the software** these problems arose in: you cannot read its source, run it, execute experiments, fit models, or call any reference tool. You have **only** the prose and mathematics written in this document. Everything you need to attack a problem is therefore stated here from first principles — and where it is not, you should reconstruct it from the definitions given, state your assumptions explicitly, and proceed.

Because you cannot run anything, **do not propose empirical procedures as solutions** ("fit it and check", "benchmark against tool X", "tune until it passes"). We are not asking you to verify against data. We are asking for **mathematics**: exact derivations, closed forms, proofs of correctness, characterizations of when something is/ isn't identifiable, optimal or provably-exact algorithms (described as mathematics and pseudocode, not as code in any language), counterexamples, and impossibility results. Where a quantity is currently computed by a finite-difference approximation, we want the **exact analytic object** that finite differences converge to, plus a proof of equivalence. Where two computations are supposed to agree, we want a **proof** that they do (or a precise characterization of the regime in which they cannot, and the corrected object).

Be maximally ambitious. For each problem, prefer the **most general** correct result over a special-case patch: derive the general operator/identity/criterion, prove it reduces to the known special case, and prove it extends correctly to the regime where the current approach fails. When a problem admits a "make-it-self-consistent" reformulation that removes a pathology entirely, prefer that and prove its equivalence to the intended semantics. State every assumption; flag every place the description under-determines the answer and give the answer for each branch.

## The setting (shared context for all problems)

The software is an engine for **penalized maximum-likelihood / generalized additive models (GAMs)** and several generalizations (location–scale / distributional regression, survival/accelerated-failure-time models, multinomial models, and Riemannian-manifold-valued and compositional responses). The recurring mathematical machinery is:

- **Penalized likelihood.** For a response `y` and a design that maps coefficients `β ∈ R^p` to a linear predictor `η = Xβ` (possibly through several additive smooth terms and several distributional parameters), one minimizes a penalized negative log-likelihood
  `F(β, θ) = −ℓ(β) + ½ βᵀ S_λ β`,
  where `S_λ = Σ_k λ_k S_k` is a sum of fixed symmetric positive-semidefinite **penalty matrices** `S_k` scaled by nonnegative **smoothing parameters** `λ_k`. Each `S_k` typically has a nontrivial null space (the "wiggliness" penalties annihilate the unpenalized polynomial/affine part of a smooth).

- **The inner problem** (for fixed hyperparameters) is a smooth convex-ish optimization solved by penalized iteratively-reweighted least squares / Newton; at the optimum `β̂(θ)` the **penalized Hessian** is `H = Xᵀ W X + S_λ` (with `W` the GLM working weights, `W = diag(p)−ppᵀ` for multinomial, etc.), and `β̂` depends on `θ` implicitly through the stationarity condition `∂F/∂β(β̂, θ) = 0`.

- **The outer problem** selects the hyperparameters `θ` (the log-smoothing-parameters `ρ = log λ`, and any nonlinear basis/kernel parameters such as a spatial length-scale `κ`, written `ψ = log κ`) by maximizing a **Laplace-approximate marginal likelihood (LAML / REML)** criterion. This criterion contains a **log-determinant** term of the penalized Hessian over an appropriate subspace; its gradient in `θ` is where several of the problems below live. **Correctness of this criterion's gradient is load-bearing**: a wrong gradient does not merely slow convergence, it moves the selected smoothing parameters and so silently biases every fitted model (over- or under-smoothing). This is why we want exact analytic objects with proofs, not approximations.

- **Identifiability / gauge.** Many sub-models carry exact invariance groups (a sum-to-zero gauge in multinomial; a multiplicative warp↔scale degeneracy in transformation survival models; an affine null space in spline penalties; conditional-centering in score models). A correct fit must break each gauge with a **minimal identifying constraint that does not bias** the estimand. Several problems ask for the precise gauge group and the unique minimal correction.

The three technical sections that follow are self-contained. Read each section's notation block first.


---

# REML / LAML criterion–gradient problems

This document poses three open mathematical problems arising in the
smoothing-parameter selection criterion of a penalized generalized additive
model (GAM). The intended solver has **no access to any code base and cannot
run anything**: every object is defined here from first principles, and every
question is framed as a derivation / proof / closed-form / optimal-algorithm
ask, not an implementation or empirical task. Be as general and rigorous as
possible; where a "fix" is requested, give the provably-correct one.

---

## 0. Common setup and notation

We define once, here, every object used in the three problems.

### 0.1 The penalized model

We observe data and fit a vector of **coefficients** $\beta \in \mathbb{R}^{p}$.
There is a **design matrix** $X \in \mathbb{R}^{n \times p}$ ($n$ observations,
$p$ coefficients) and a **linear predictor** $\eta = X\beta \in \mathbb{R}^n$.
A scalar **log-likelihood** $\ell(\beta)$ (e.g. Gaussian, Bernoulli/binomial
with a logit or probit link, Poisson, …) depends on $\beta$ only through
$\eta$. For an exponential-family GLM with link $g$, mean
$\mu_i = g^{-1}(\eta_i)$, and (possibly) a dispersion $\phi$,

$$
\ell(\beta) \;=\; \sum_{i=1}^n \frac{y_i \theta_i - b(\theta_i)}{\phi} + c(y_i,\phi),
\qquad \theta_i = \theta(\eta_i),
$$

with $\nabla_\beta \ell = X^\top u$ for a **score vector** $u \in \mathbb{R}^n$
($u_i = \partial \ell / \partial \eta_i$), and Hessian
$\nabla^2_\beta \ell = -X^\top W X$ where
$W = \mathrm{diag}(w_1,\dots,w_n)$, $w_i = -\partial^2 \ell/\partial\eta_i^2$
(the IRLS / Fisher weights; $W \succeq 0$ in the canonical/Fisher case).
For Gaussian identity, $w_i \equiv 1/\sigma^2$ and $W$ is constant; for a
nonlinear link, $w_i$ depends on $\beta$ through $\eta_i$, so $W = W(\beta)$.

### 0.2 The smoothing penalty and hyperparameters

Smoothness is imposed by a quadratic penalty. There are $m$ symmetric
**positive-semidefinite penalty matrices** $S_k \in \mathbb{R}^{p\times p}$,
$S_k = S_k^\top \succeq 0$, each typically **rank-deficient**
($\operatorname{rank} S_k < p$; e.g. a second-difference / curvature penalty
annihilates the constant and linear polynomials, which span its null space).
The combined penalty is

$$
S_\lambda \;=\; \sum_{k=1}^m \lambda_k\, S_k, \qquad \lambda_k > 0,
\qquad S_\lambda = S_\lambda^\top \succeq 0 .
$$

The **hyperparameters** are collected in a vector
$\theta = (\rho, \psi) \in \mathbb{R}^{K}$:

* $\rho = (\rho_1,\dots,\rho_m) = \log\lambda$, i.e.
  $\lambda_k = e^{\rho_k}$. These are **multiplicative scale** parameters:
  each $\rho_k$ scales its block $S_k$ but does **not** change the matrices
  $S_k$ themselves. Hence
  $\partial S_\lambda / \partial \rho_k = \lambda_k S_k = e^{\rho_k} S_k$, and
  the column space $\operatorname{range}(S_\lambda)$ is **invariant** under
  changes of $\rho$ (a sum of fixed PSD subspaces scaled by positive weights
  spans the same space for all positive weights, generically).

* $\psi = (\psi_1,\dots) = \log\kappa$, where $\kappa$ is a **kernel /
  length-scale (shape) parameter** of a spatial smoother (e.g. the
  correlation range of a Matérn field, or the scale of a Duchon / thin-plate
  spline). Crucially, $\kappa$ enters the **basis and the penalty matrices
  nonlinearly**: the $S_k$ themselves are functions $S_k(\psi)$ (a Matérn or
  Duchon penalty is assembled from $\kappa$-dependent Gram / operator
  matrices). Therefore
  $\partial S_\lambda/\partial \psi_j = \sum_k \lambda_k\, \partial S_k/\partial\psi_j
  =: S_{\lambda,\psi_j} \neq 0$, and — unlike the $\rho$ case — this
  derivative is **not** a re-scaling of $S_\lambda$: it can change the
  **column space** $\operatorname{range}(S_\lambda)$ as $\psi$ varies. We say
  the penalty's range **rotates** with $\psi$.

We write a generic hyperparameter coordinate as $\tau \in \{\rho_k\} \cup \{\psi_j\}$.

### 0.3 The inner problem and the penalized Hessian

For fixed $\theta$, the **inner problem** fits $\beta$ by maximizing the
penalized log-likelihood (equivalently minimizing the penalized deviance):

$$
\hat\beta(\theta) \;=\; \arg\min_{\beta}\; F(\beta,\theta),
\qquad
F(\beta,\theta) \;=\; -\,\ell(\beta) \;+\; \tfrac12\, \beta^\top S_\lambda(\theta)\, \beta .
$$

Its stationarity condition (the implicit definition of $\hat\beta$) is

$$
\nabla_\beta F(\hat\beta,\theta) \;=\; -\,X^\top u(\hat\beta) \;+\; S_\lambda \hat\beta \;=\; 0 .
\tag{IFT}
$$

The **penalized (Fisher) Hessian** of the inner objective is

$$
H(\beta,\theta) \;=\; \nabla^2_\beta F \;=\; X^\top W(\beta)\, X \;+\; S_\lambda(\theta)
\;=\; \underbrace{X^\top W X}_{=:\,H_\ell\ \text{(unpenalized info)}} + \; S_\lambda .
$$

At the inner optimum we write $H := H(\hat\beta(\theta),\theta)$. In the
Gaussian-identity case $H_\ell = X^\top X/\sigma^2$ is constant; for a
nonlinear link $H_\ell$ depends on $\theta$ through $\hat\beta(\theta)$.

### 0.4 The marginal-likelihood (LAML / REML) criterion

Smoothing parameters are selected by maximizing a Laplace-approximate marginal
likelihood (LAML), equivalently minimizing the REML/LAML objective

$$
\mathcal{V}(\theta)
\;=\; -\,\ell\!\big(\hat\beta(\theta)\big)
\;+\; \tfrac12\,\hat\beta(\theta)^\top S_\lambda(\theta)\,\hat\beta(\theta)
\;+\; \tfrac12\,\log\big|H(\theta)\big|_{+}
\;-\; \tfrac12\,\log\big|S_\lambda(\theta)\big|_{+}
\;+\; (\text{const}),
\tag{V}
$$

where $|\cdot|_+$ denotes a **generalized (pseudo) determinant**, defined next,
needed because both $S_\lambda$ and (in the rank-deficient regime) $H$ may be
singular. The smoothing parameters are
$\hat\theta = \arg\min_\theta \mathcal V(\theta)$. The **first two terms** are
the penalized fit at $\hat\beta$; the $\tfrac12\log|H|_+$ term is the Laplace
"Occam factor"; the $-\tfrac12\log|S_\lambda|_+$ term is the prior
normalization. The two problems below concern (1) the **log-determinant
term(s)** and their $\theta$-gradient, and (2)–(3) the **outer-Hessian /
coefficient-sensitivity** structure of $\mathcal V$.

### 0.5 Generalized log-determinant on a subspace

For a symmetric $A = A^\top \in \mathbb{R}^{p\times p}$ with eigendecomposition
$A = \sum_i \sigma_i q_i q_i^\top$, the **pseudo-log-determinant** on the
positive eigenspace is $\log|A|_+ = \sum_{\sigma_i > 0} \log \sigma_i$.

There are two distinct ways the criterion (V) realizes the
$\tfrac12\log|H|_+$ term in the **rank-deficient** regime, and the discrepancy
between them is the subject of Problem 1:

* **Full identifiable subspace.** Let $T(\theta) = \operatorname{range}(H+S_\lambda)$ —
  but in the convention used here $H$ already contains $S_\lambda$, so we mean
  the column space of the *total* penalized Hessian, the directions in which
  the posterior is proper. Then $\log|H|_+$ is the pseudo-log-determinant over
  $T$. The derivative of $\log|H|_+$ over the **full** identified subspace with
  respect to any perturbation direction $\dot H$ is captured by the standard
  generalized-determinant identity (Section 0.6).

* **Penalty-range projection ($U_S$ kernel).** Let
  $U_S(\theta) \in \mathbb{R}^{p\times r}$ be an **orthonormal basis of**
  $\operatorname{range}\big(S_\lambda(\theta)\big)$,
  $U_S^\top U_S = I_r$, $r = \operatorname{rank} S_\lambda$. Define the
  orthogonal **projector onto the penalty range**
  $P(\theta) = U_S U_S^\top$. The criterion realizes the determinant term as

  $$
  \tfrac12 \log\big| U_S(\theta)^\top\, H(\theta)\, U_S(\theta) \big|_{+}
  \;=\; \tfrac12 \log\big| H_{\mathrm{proj}}(\theta) \big|,
  \qquad
  H_{\mathrm{proj}} := U_S^\top H\, U_S \in \mathbb{R}^{r\times r}.
  \tag{LDP}
  $$

  Its analytic $\tau$-derivative is computed by a **fixed-$U_S$ trace kernel**:
  with $H_{\mathrm{proj}}^{-1}$ precomputed and
  $K := U_S\, H_{\mathrm{proj}}^{-1}\, U_S^\top$ (a $p\times p$ matrix of rank
  $r$, equal to the range($S_\lambda$) block of the full pseudo-inverse
  $(H)_+^{\,-1}$), the code uses

  $$
  \frac{d}{d\tau}\,\tfrac12\log|H_{\mathrm{proj}}|
  \;\stackrel{\text{(code)}}{=}\;
  \tfrac12\,\operatorname{tr}\!\big(K\, \dot H\big)
  \;=\; \tfrac12\,\operatorname{tr}\!\big(H_{\mathrm{proj}}^{-1}\, U_S^\top \dot H\, U_S\big),
  \qquad \dot H := \partial H/\partial\tau,
  \tag{KER}
  $$

  i.e. it differentiates only $H$ inside the projection and **holds $U_S$
  fixed**.

### 0.6 The standard (Jacobi) determinant derivative

For a symmetric, nonsingular $A(\tau)$,
$\dfrac{d}{d\tau}\log|A| = \operatorname{tr}\!\big(A^{-1}\dot A\big)$. For a
PSD $A(\tau)$ of constant rank with column space $\operatorname{range}(A)$
**fixed** in $\tau$, the generalized version is
$\dfrac{d}{d\tau}\log|A|_+ = \operatorname{tr}\!\big(A_+^{-1}\dot A\big)$,
where $A_+^{-1}$ is the Moore–Penrose pseudo-inverse, **provided** the
perturbation $\dot A$ leaves $\operatorname{range}(A)$ invariant (no
eigenvector rotation across the zero/positive boundary). When the column space
*moves*, this formula is incomplete: there is an extra term from the rotation
of the eigenprojectors, which the next problem makes precise.

---

## Problem 1 — Exact gradient of the projected log-determinant under a moving penalty range

### 1.1 The phenomenon

Consider the projected determinant term (LDP) and its code-derivative (KER).
Empirically (by finite differences that **rebuild** $U_S$ at $\psi \pm h$), the
analytic derivative (KER) is correct for $\rho$ in the **Gaussian** case but
**wrong** in two situations for general (non-Gaussian) GLMs:

1. **$\psi$-derivative (moving subspace).** Because $S_{\lambda,\psi_j} \neq 0$
   is not a re-scaling of $S_\lambda$, the range
   $\operatorname{range}(S_\lambda(\psi))$ — and therefore the orthonormal basis
   $U_S(\psi)$ and the projector $P(\psi) = U_S U_S^\top$ — **rotate** with
   $\psi$. The true derivative of (LDP) contains a *moving-subspace term*
   $\propto \partial U_S/\partial\psi_j$ (equivalently $\partial P/\partial\psi_j$)
   that (KER) drops by holding $U_S$ fixed. A finite-difference evaluation
   $\big[\tfrac12\log|U_S(\psi+h)^\top H U_S(\psi+h)| - (\psi-h)\big]/(2h)$
   includes this term; the analytic (KER) does not. The magnitude of the
   dropped term grows with $\|S_{\lambda,\psi}\|$ and the disagreement is
   observed to blow up by several orders of magnitude.

2. **$\rho$-derivative (GLM null-space leakage).** Even in the $\rho$ direction,
   where $\operatorname{range}(S_\lambda)$ is invariant, the analytic
   derivative is observed to be *sign-flipped / wrong* for non-Gaussian GLMs.
   The structural reason: for a nonlinear link, $\dot H$ in the $\rho$-direction
   is **not** simply $\lambda_k S_k$. There is an implicit-function-theorem
   contribution because $\hat\beta(\theta)$, hence $W(\hat\beta)$, hence
   $H_\ell = X^\top W X$, depends on $\rho$. That IFT contribution
   $D_\beta H[\dot\beta] = X^\top \mathrm{diag}(c \odot X\dot\beta) X$ (with
   $c$ a vector of third-derivative link weights, $\dot\beta = \partial\hat\beta/\partial\rho_k$)
   has support that **leaks onto $\operatorname{null}(S_\lambda)$** — e.g. the
   intercept column $X_{:,0}=\mathbf 1_n$ lies in $\operatorname{null}(S_\lambda)$.
   The kernel (KER) only includes $U_S^\top \dot H\, U_S$ (the penalty-range
   block) and either drops or mis-signs the null-space-coupled portion that the
   *true* derivative of the full-subspace logdet retains. For Gaussian identity
   $c=0$, so this leakage vanishes — consistent with Gaussian passing.

### 1.2 The asks

**(1.2a) Exact analytic derivative with a moving projector.** Let
$U_S(\theta) \in \mathbb{R}^{p\times r}$ be an orthonormal basis of
$\operatorname{range}(S_\lambda(\theta))$, with orthogonal projector
$P(\theta) = U_S U_S^\top$ (note $P$ is **basis-independent** even though
$U_S$ is only defined up to an $r\times r$ orthogonal gauge). Define the
scalar criterion
$\Phi(\theta) = \log\big| U_S(\theta)^\top H(\theta)\, U_S(\theta)\big|$.
Derive the **exact** total derivative $\partial\Phi/\partial\tau$ for an
arbitrary $\theta$-coordinate $\tau$, fully accounting for the $\theta$-
dependence of $P$ and of $H$. In particular:

  - Express the answer in a manifestly **gauge-invariant** form (independent of
    the arbitrary choice of orthonormal basis $U_S$ within the range), i.e. in
    terms of $P$, $\dot P := \partial P/\partial\tau$, $H$, $\dot H$, and the
    appropriate pseudo-inverse — **not** in terms of $\dot U_S$ alone.
  - Give the closed-form expression for the moving-subspace correction
    $\dot P$ in terms of $S_\lambda$, $\dot S_\lambda := \partial S_\lambda/\partial\tau$,
    and $S_\lambda^{+}$ (Moore–Penrose pseudo-inverse). (Hint: $P$ is the
    spectral projector of $S_\lambda$ onto its strictly-positive eigenvalues;
    differentiating $P = P^2$, $PS_\lambda = S_\lambda$, and using the
    Sylvester/resolvent identity for projector perturbation, gives
    $\dot P$ in terms of $S_\lambda^{+}\dot S_\lambda(I-P)$ and its transpose.
    State and prove the exact identity.)
  - Prove that your $\partial\Phi/\partial\tau$ equals the finite-difference
    limit $\lim_{h\to0}\big[\Phi(\theta+h e_\tau)-\Phi(\theta-h e_\tau)\big]/(2h)$,
    treating with care the case where an eigenvalue of $S_\lambda$ crosses the
    zero threshold (rank change) — characterize exactly when $\Phi$ is
    differentiable and what happens at a rank-change locus.

**(1.2b) The $\rho$-direction GLM correction.** Specialize $\tau=\rho_k$ (so
$\dot P = 0$ because the range is $\rho$-invariant — *prove* this invariance,
or characterize the measure-zero exceptions). Derive the exact
$\partial\Phi/\partial\rho_k$ for a non-Gaussian GLM, where
$\dot H = \lambda_k S_k + D_\beta H[\dot\beta]$ and
$\dot\beta = \partial\hat\beta/\partial\rho_k = -H^{-1}\partial(\nabla_\beta F)/\partial\rho_k
= -H^{-1}(\lambda_k S_k)\hat\beta$ from (IFT). Show **precisely** which part of
the resulting trace the fixed-$U_S$ kernel (KER) drops or mis-signs, and write
the corrected closed form. Establish whether the correct $\rho$-derivative of
the **full identifiable-subspace** logdet $\log|H|_+$ (over
$\operatorname{range}(H)$, not over $\operatorname{range}(S_\lambda)$) is given
cleanly by $\operatorname{tr}(H_+^{-1}\dot H)$ and why that differs from the
projected-kernel value $\operatorname{tr}(K\dot H)$ with
$K = U_S H_{\mathrm{proj}}^{-1} U_S^\top$.

**(1.2c) Reformulation equivalence (the decisive theoretical question).**
Standard practice (cf. mgcv's generalized-determinant derivative) is to
realize the Laplace term as $\tfrac12\log|H|_+$ over
$\operatorname{range}(H)$ — the **full identifiable subspace of the total
penalized Hessian** — whose derivative
$\tfrac12\operatorname{tr}(H_+^{-1}\dot H)$ is valid for **any** perturbation
direction $\dot H$ (including ones with null$(S_\lambda)$ support), because
$\operatorname{range}(H)$ does not collapse under the inner-optimum curvature.
Prove or refute:

  > For a penalized GLM at the inner optimum $\hat\beta(\theta)$, the
  > projected-range criterion $\tfrac12\log|U_S^\top H U_S|$ and the
  > full-subspace criterion $\tfrac12\log|H|_+$ define the **same** LAML
  > objective up to a $\theta$-independent additive constant; consequently the
  > correct gradient of the projected criterion **equals**
  > $\tfrac12\operatorname{tr}(H_+^{-1}\dot H)$ for every $\theta$-direction
  > (both $\rho$ and $\psi$), and the moving-subspace pathology of (KER) is an
  > artifact of differentiating the *wrong* (projected) realization rather than
  > a genuine property of the LAML criterion.

State the exact conditions on $H$, $S_\lambda$, and their null spaces
($\operatorname{null}(H)$ vs. $\operatorname{null}(S_\lambda)$, the
unpenalized-but-identified directions, and the genuinely unidentified
directions $\operatorname{null}(X)\cap\operatorname{null}(S_\lambda)$) under
which the two realizations coincide, and where they legitimately differ (e.g.
a likelihood-identified-but-unpenalized direction contributes to $\log|H|_+$
but lies outside $\operatorname{range}(S_\lambda)$, so it is **dropped** by the
projected form). Give the **general, provably-correct criterion** to use, and
its exact gradient valid for arbitrary $\theta$ (handling both the $\rho$ scale
and the $\psi$ moving-range directions in one formula), together with the
pseudo-inverse / trace structure needed to evaluate it without an
eigenbasis-rotation term. (Relate, where it clarifies, to the dual situations:
one fix that moved a kernel from $\operatorname{range}(S_\lambda)$ to
$\operatorname{range}(H+S_\lambda)$ to recover unpenalized identified
curvature, and one that replaced a discontinuous $\delta$-ridged
$\log|H+S_\lambda+\delta I|$ value — which desynced from a
$\operatorname{tr}((H+S_\lambda)^{-1}\cdot)$ gradient — with a strict
positive-eigenspace pseudo-logdet that is $C^\infty$ in $\theta$. Both are
instances of "make the value and the gradient differentiate the **same**
$C^\infty$ subspace.")

---

## Problem 2 — Profiled outer-Hessian θ-HVP by implicit differentiation

### 2.1 Setup

Recall the profiled (concentrated) outer objective
$g(\theta) := \mathcal V(\theta)$ from (V), where the inner solution
$\hat\beta(\theta)$ is defined implicitly by stationarity (IFT):
$\nabla_\beta F(\hat\beta,\theta)=0$. Thus $g$ is a function of $\theta$ alone,
but every evaluation of its derivatives must differentiate **through** the
inner solve. The smoothing parameters are chosen by a second-order
(trust-region / Newton) outer optimizer that needs the **outer gradient**
$\nabla_\theta g \in \mathbb{R}^K$ and the **outer Hessian**
$\nabla^2_\theta g \in \mathbb{R}^{K\times K}$, $K = \dim\theta$.

The status quo assembles $\nabla^2_\theta g$ **coordinate-pair by
coordinate-pair**: for each pair $(\tau_a,\tau_b)$ it forms third/fourth
contractions, at cost scaling like $K^2 \times n \times (\text{per-row
higher-derivative work}) \times (\text{trace rank})$. This is the dominant
cost and explodes even for small $K$ (e.g. $\dim\psi = 2$).

### 2.2 Implicit-function-theorem building blocks

From (IFT), differentiating $\nabla_\beta F(\hat\beta(\theta),\theta)=0$:

$$
H\,\frac{\partial\hat\beta}{\partial\tau} + \frac{\partial \nabla_\beta F}{\partial\tau}\Big|_{\beta\,\text{fixed}} = 0
\;\;\Longrightarrow\;\;
\dot\beta_\tau := \frac{\partial\hat\beta}{\partial\tau}
= -\,H^{-1}\, \frac{\partial^2 F}{\partial\beta\,\partial\tau},
\tag{S1}
$$

where $H = \nabla^2_\beta F$ as in §0.3 and
$\partial^2 F/\partial\beta\,\partial\tau \in \mathbb{R}^p$ is the mixed
partial holding $\beta$ fixed (for $\tau=\rho_k$ this is
$\lambda_k S_k\hat\beta$; for $\tau=\psi_j$ it is
$\tfrac12\,\partial(\,\hat\beta^\top S_\lambda\hat\beta)/\partial\psi_j$'s
gradient piece, i.e. $S_{\lambda,\psi_j}\hat\beta$). For a direction
$v\in\mathbb{R}^K$ in hyperparameter space, define the **inner-sensitivity HVP**

$$
\dot\beta[v] \;=\; \sum_\tau v_\tau\,\dot\beta_\tau \;=\; -\,H^{-1} B v,
\qquad B := \frac{\partial^2 F}{\partial\beta\,\partial\theta} \in \mathbb{R}^{p\times K}.
\tag{S2}
$$

### 2.3 The asks

**(2.3a) Exact matrix-free outer Hessian-vector product.** Derive the exact
$\nabla^2_\theta g(\theta)\, v$ as an operator applied to an arbitrary
$v\in\mathbb{R}^K$, using only:

  - the inner solve operator $w\mapsto H^{-1}w$ (one linear solve with the
    already-factored penalized Hessian),
  - directional second and **third** derivatives of the inner objective $F$
    in $\beta$ and in $\theta$ (specify exactly which: the third-order tensors
    $\partial^3 F/\partial\beta^3[\cdot,\cdot]$,
    $\partial^3 F/\partial\beta^2\partial\theta$,
    $\partial^3 F/\partial\beta\,\partial\theta^2$, etc., **applied as
    operators** — never materialized), and
  - directional first and second derivatives of the log-determinant terms
    $\tfrac12\log|H|_+$ and $-\tfrac12\log|S_\lambda|_+$.

Give the full expansion. The total derivative of $g$ obeys (envelope theorem
at the inner optimum) $\nabla_\theta g = \partial_\theta \mathcal V|_{\hat\beta}$
for the fit terms, plus the explicit log-det $\theta$-gradients; for the
**Hessian** the $\hat\beta$-dependence no longer drops and the chain rule must
carry $\dot\beta[v]$ through. Concretely, derive the components:

  1. Differentiate the penalized-fit term
     $-\ell(\hat\beta)+\tfrac12\hat\beta^\top S_\lambda\hat\beta$ twice in
     $\theta$, using (S1)–(S2) and the third derivative of $F$.
  2. Differentiate $\tfrac12\log|H(\theta)|_+$ twice in $\theta$, where
     $H = X^\top W(\hat\beta)X + S_\lambda$ depends on $\theta$ both explicitly
     (through $S_\lambda$) and implicitly (through $\hat\beta$). You will need
     $\partial H/\partial\tau = S_{\lambda,\tau} + D_\beta H[\dot\beta_\tau]$
     and the log-det second directional derivative
     $\tfrac{d^2}{d\tau\,d\tau'}\log|H|_+
     = \operatorname{tr}\!\big(H_+^{-1}\ddot H\big) - \operatorname{tr}\!\big(H_+^{-1}\dot H_\tau H_+^{-1}\dot H_{\tau'}\big)$
     (state the rank-deficient generalized-determinant version precisely,
     consistent with Problem 1).
  3. Differentiate $-\tfrac12\log|S_\lambda(\theta)|_+$ twice (this term has no
     $\hat\beta$ dependence but has the moving-range subtlety from Problem 1 in
     the $\psi$ directions).

Assemble these into a single operator $v\mapsto \nabla^2_\theta g\,v$ that uses
$O(1)$ inner solves with $H$ per HVP and **never** materializes a $K\times K$
matrix or a $p\times p$ third-derivative tensor.

**(2.3b) Equivalence proof.** Prove that the operator from (2.3a), when probed
along the coordinate directions $\{e_a\}$, reproduces **exactly** the dense
pairwise outer Hessian $[\nabla^2_\theta g]_{ab}$ that the status-quo
coordinate-pair assembly computes (i.e. the HVP is the true Hessian, not a
Gauss–Newton or otherwise dropped-term approximation). State precisely which,
if any, third-order terms the pair-assembly drops or keeps, so the proof is an
identity, not an approximation.

**(2.3c) Symmetry.** Characterize the conditions under which
$\nabla^2_\theta g$ is exactly symmetric (it must be, as a Hessian of a
$C^2$ scalar). Identify which terms in the operator are *manifestly* symmetric
(e.g. the $\operatorname{tr}(H_+^{-1}\dot H_\tau H_+^{-1}\dot H_{\tau'})$ piece)
and which acquire numerical asymmetry from the finite-precision $H^{-1}$ solve
or from a non-self-adjoint application order. Give an exact
symmetry-preserving formulation (e.g. an explicitly symmetric quadratic-form
realization $v^\top \nabla^2_\theta g\, v$ from which the operator is recovered
by polarization), and prove it is self-adjoint at the operator level.

**(2.3d) Self-checking correctness certificate.** Because a subtly wrong
profiled $\theta$-derivative does **not** crash — it silently biases the REML
criterion and causes systematic over- or under-smoothing — provide a **scalar
identity** (a quantity computable by two independent routes that must agree to
machine precision when the derivation is correct) usable as a correctness
certificate **without** finite differences. Examples to derive rigorously:
(i) the directional consistency $v^\top(\nabla^2_\theta g)\,v$ vs. the second
derivative along a 1-D probe obtained by differentiating the analytic gradient
$\nabla_\theta g(\theta+t v)$ once more analytically; (ii) a Schur-complement
identity linking the profiled outer Hessian to a block of the **joint**
$(\beta,\theta)$ Hessian of $\mathcal V$ (the profiled Hessian is the Schur
complement of the $\beta\beta$ block); state and prove the exact relation

$$
\nabla^2_\theta g
\;=\; \mathcal V_{\theta\theta} - \mathcal V_{\theta\beta}\,\mathcal V_{\beta\beta}^{-1}\,\mathcal V_{\beta\theta},
$$

with the blocks defined from the joint $\mathcal V(\beta,\theta)$ (before
profiling), and show how matching the two routes certifies the HVP.

---

## Problem 3 — Cone of influence: exact localized coefficient-sensitivity

### 3.1 Setup

From (S1), the sensitivity of the fitted coefficients to a single smoothing
parameter $\tau_i$ is
$\dot\beta_{\tau_i} = -H^{-1}\,b_i$ with $b_i := \partial^2 F/\partial\beta\,\partial\tau_i$.
In a model with many smooth terms, $H$ has block / sparsity structure (each
penalty $S_k$ acts on a contiguous coefficient block; $X^\top W X$ couples
blocks only where their basis functions have overlapping support or share a
common factor in a tensor-product term). Most of $\hat\beta$ is **insensitive**
to a move in $\tau_i$: the vector $b_i$ is supported on the block(s) associated
with $\tau_i$, and $H^{-1}b_i$ spreads that support only along the coupling
graph of $H$. Recomputing the *full* $\dot\beta_{\tau_i}$ when only one
$\tau_i$ moves is wasteful.

### 3.2 The asks

**(3.2a) Exact cone of influence.** Model $H$ as the adjacency/weight matrix of
a graph on coefficient blocks: blocks $a,b$ are *coupled* iff the $(a,b)$
off-diagonal block of $H$ is nonzero. Let $\mathcal R(b_i)$ be the set of
blocks on which $b_i$ is structurally nonzero. Characterize **exactly** the set
$\mathcal C_i$ of coefficient blocks on which $\dot\beta_{\tau_i}=-H^{-1}b_i$
is structurally nonzero (the *cone of influence* of $\tau_i$), in terms of the
graph of $H$ and the support $\mathcal R(b_i)$. Note that $H^{-1}$ is generally
**dense even when $H$ is sparse** (the inverse of a sparse matrix fills in), so
naively $\mathcal C_i$ is everything. State precisely the structural conditions
under which the cone is **strictly smaller** than the full coefficient vector —
e.g. when $H$ is block-diagonal / block-tridiagonal / has a separator
(graph-partition) structure, or when $b_i$ lies in an $H$-invariant subspace —
and give the exact $\mathcal C_i$ in each case. Distinguish "structurally zero"
(exactly zero by sparsity, for all numerical values) from "numerically
negligible" (small but nonzero); **only the former** licenses an exact
localized solve.

**(3.2b) Exactness of the block-restricted solve.** Suppose we restrict the IFT
solve to a block-submatrix $H_{\mathcal C\mathcal C}$ (the principal submatrix on
a candidate cone $\mathcal C \supseteq \mathcal R(b_i)$) and solve
$H_{\mathcal C\mathcal C}\,(\dot\beta_{\tau_i})_{\mathcal C} = -(b_i)_{\mathcal C}$,
leaving the complement at its cached value. Derive the **exact** condition on
$H$ and $\mathcal C$ under which this localized solve equals the global solve
$-H^{-1}b_i$ restricted to $\mathcal C$ (i.e. **no approximation**, not "small
error"). Show this holds iff $\mathcal C$ is *closed* under the coupling graph
relative to the support of $b_i$ — precisely, iff the complement
$\bar{\mathcal C}$ is decoupled from the right-hand side through the Schur
complement: $H_{\bar{\mathcal C}\mathcal C}$ acting on $(b_i)_{\mathcal C}$
produces no fill that re-enters $\mathcal C$. State the exact algebraic
identity (Schur-complement / nested-dissection form) and prove the
equivalence. Characterize the *minimal* exact cone $\mathcal C_i^\star$ and give
an optimal algorithm (graph reachability on the coupling graph) to compute it,
with its complexity.

**(3.2c) Row-attribution of a θ-HVP (strict extension of Problem 2).** A
$\theta$-HVP $\nabla^2_\theta g\,v$ (Problem 2) ultimately reduces to traces and
quadratic forms involving $\dot\beta[v] = -H^{-1}Bv$ and the per-observation
contributions $X_{i,:}$, $w_i$, and the higher link derivatives $c_i$. Derive
the **exact row-attribution decomposition**: express the scalar
$v^\top(\nabla^2_\theta g)\,v$ (or, for direction $i$, the component along
$e_i$) as a sum over observations $\sum_{j=1}^n \chi_j$ with each $\chi_j$ a
closed-form per-row contribution, so that one can say *exactly* which
observations contribute to the outer-Hessian action in a given hyperparameter
direction. State which rows have $\chi_j \equiv 0$ structurally (e.g. rows
whose design support $X_{j,:}$ lies entirely outside the cone of influence
$\mathcal C_i$, or rows with $w_j$ or $c_j$ vanishing for Gaussian identity).
Prove the decomposition is exact (sums to the full HVP) and that the
structurally-zero rows can be skipped **without** any tolerance — i.e. the
localized, row-pruned HVP is bit-for-bit equal to the dense HVP, characterized
purely by the sparsity of $X$, the cone of influence, and the link-derivative
support. This unifies Problems 2 and 3: the cone of influence (3.2a–b) is
exactly the structural support that makes the row-attributed HVP (3.2c)
localizable without approximation.

---

### Cross-cutting requirements

* All three problems concern a **REML/LAML criterion**: a wrong derivative does
  not error, it biases smoothing-parameter selection. Therefore every result
  must be **exact** (no first-order or Gauss–Newton truncation) and accompanied
  by a proof of equality to the finite-difference / dense ground truth, plus,
  where requested, a finite-difference-free self-consistency certificate.
* Prefer the most **general** formulation: arbitrary exponential-family
  likelihood with a smooth link (treat Gaussian identity as the degenerate
  $c=0$, constant-$W$ special case), arbitrary number and rank of PSD penalty
  blocks, and both multiplicative ($\rho$) and shape ($\psi$) hyperparameters
  including the moving-range case.
* State all results in **gauge-invariant** form where a subspace basis is
  otherwise arbitrary (e.g. use projectors $P=U_SU_S^\top$ and pseudo-inverses,
  not a particular $U_S$), so the answer does not depend on an arbitrary
  orthonormal-basis choice.


---

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


---

# Manifold geometry and Matérn penalty structure

Two self-contained mathematical problems. Each is posed from first principles for a
reader with no access to any codebase and no ability to execute code. Every object is
defined before it is used. The asks are pure derivations, proofs, and closed forms.

---

## Notation and standing conventions

- `ℝ` is the field of real numbers; `ℝⁿ` is Euclidean `n`-space with the standard inner
  product `⟨a,b⟩ = aᵀb` and norm `‖a‖ = √(aᵀa)`.
- For a real matrix `A`, `Aᵀ` is its transpose, `‖A‖_F = √(tr(AᵀA)) = √(Σ_{ij} A_{ij}²)`
  is its Frobenius norm, and `σ_i(A)` denotes the `i`-th singular value (in non-increasing
  order). `I_k` is the `k×k` identity.
- A matrix `Y ∈ ℝ^{n×p}` is *orthonormal* (a *Stiefel frame*) when `YᵀY = I_p`; its column
  span `span(Y) ⊆ ℝⁿ` is a `p`-dimensional linear subspace, i.e. a point of the
  Grassmann manifold defined below.
- `arccos: [−1,1] → [0,π]`, `arcsin: [−1,1] → [−π/2, π/2]`, `atan2(y,x)` is the
  two-argument arctangent returning the polar angle of `(x,y)` in `(−π, π]`.
- "Machine precision" / "ulp" refer to IEEE-754 binary64: unit roundoff
  `u = 2^{−53} ≈ 1.11·10^{−16}`, and a *1-ulp error in* `s ∈ (0,1]` means an absolute
  perturbation of size up to `½·ulp(s) ≈ u·s` (bounded above by `u` for `s ≤ 1`).

---

## Problem 1 — Grassmann geodesic distance vs. the projector (chordal) metric near the cut locus

### 1.1 The Grassmann manifold

Fix integers `0 < p ≤ n`. The **Grassmann manifold** `Gr(p, n)` is the set of all
`p`-dimensional linear subspaces of `ℝⁿ`. A subspace `𝒮 ∈ Gr(p,n)` is represented
(non-uniquely) by any orthonormal frame `Y ∈ ℝ^{n×p}`, `YᵀY = I_p`, with `span(Y) = 𝒮`.
Two frames `Y, Y'` represent the same subspace iff `Y' = YQ` for some orthogonal
`Q ∈ O(p)`. Thus `Gr(p,n) = St(p,n) / O(p)` where `St(p,n) = {Y : YᵀY = I_p}` is the
compact Stiefel manifold. `Gr(p,n)` is a smooth compact manifold of dimension `p(n−p)`.

`Gr(p,n)` carries a *canonical* Riemannian metric, induced from the standard inner
product on `ℝ^{n×p}` (equivalently, the bi-invariant metric on `O(n)` pushed down through
the quotient `O(n)/(O(p)×O(n−p))`). On the horizontal space at `Y` (tangent vectors
`H ∈ ℝ^{n×p}` with `YᵀH = 0`), the metric is `⟨H₁,H₂⟩ = tr(H₁ᵀH₂)`. This is the metric
relative to which all "geodesic distance" statements below are made.

### 1.2 Principal angles and the two competing distances

Let `𝒮 = span(Y)` and `𝒯 = span(Z)` with `Y, Z ∈ ℝ^{n×p}` orthonormal. The **principal
angles** `0 ≤ θ_1 ≤ θ_2 ≤ … ≤ θ_p ≤ π/2` between `𝒮` and `𝒯` are defined recursively by
`cos θ_k = max{ uᵀv : u ∈ 𝒮, v ∈ 𝒯, ‖u‖=‖v‖=1, u ⟂ u_j, v ⟂ v_j (j<k) }`, with
maximizers `(u_k, v_k)` the principal vectors. Equivalently, if `YᵀZ = U Σ Vᵀ` is a
singular value decomposition with singular values `σ_1 ≥ … ≥ σ_p ∈ [0,1]`, then
`σ_i = cos θ_{p+1−i}`; we write simply `σ_i = cos θ_i` after matching the orderings, so
each principal angle is `θ_i = arccos(σ_i)`. The principal angles depend only on the
subspaces, not on the chosen frames (replacing `Y → YQ₁`, `Z → ZQ₂` multiplies `YᵀZ`
by orthogonal matrices on each side, preserving its singular values).

Two distance notions arise.

**(A) Canonical geodesic (arc-length) distance.** The Riemannian distance induced by the
canonical metric of §1.1 is
```
    d_geo(𝒮, 𝒯) = √( Σ_{i=1}^{p} θ_i² ) = √( Σ_{i=1}^{p} arccos²(σ_i) ).
```
This is the length of the minimizing geodesic; the geodesic itself is `t ↦ span(Y V cos(tΘ) + W sin(tΘ))` for a suitable orthonormal `W` and `Θ = diag(θ_i)`.

**(B) Projector / chordal metric.** Each subspace has a unique orthogonal *projector*
`P_Y = Y Yᵀ ∈ ℝ^{n×n}` (symmetric, idempotent, rank `p`), independent of the frame
representative. A common, frame-free metric is the Frobenius distance between projectors,
```
    d_proj(𝒮, 𝒯) = ‖P_Y − P_Z‖_F = √2 · √( Σ_{i=1}^{p} sin²θ_i ),
```
the second equality being a standard identity (proven via
`‖P_Y − P_Z‖_F² = 2p − 2‖YᵀZ‖_F² = 2 Σ(1 − cos²θ_i) = 2 Σ sin²θ_i`). Variants in use
("chordal", "projection F-norm", `√2·sin`-type, `2·sin(θ/2)` "chord on the sphere of
projectors") all share the property that they are smooth functions of `{sinθ_i, cosθ_i}`
that **saturate** as `θ_i → π/2`, in contrast to `d_geo` which is *linear* in `θ_i`.

### 1.3 The observed discrepancy (the phenomenon to explain)

On real data (subspaces obtained as leading principal-component spans of grouped
datasets), the canonical arc-length `d_geo = √(Σ arccos²σ_i)` is computed to ≈`10^{−14}`
agreement with the analytic principal-angle formula, including pairs whose **largest
principal angle approaches `π/2`** (the *cut locus* direction). A reference library that
instead reports a **projector-based** distance `metric.dist` disagrees from `d_geo` by up
to ≈ `0.64` on exactly those near-`π/2` pairs. The disagreement is not roundoff: it is
the structural gap between (A) and (B).

### 1.4 Asks

**(1a) Closed-form gap and identification of the true geodesic distance.** Define the
per-angle discrepancy and prove the exact relationship between the canonical geodesic
distance and the chordal/projector family. Specifically:

  - Show that for a single principal angle `θ ∈ [0, π/2]`, the canonical contribution is
    `θ` while the chordal contribution is `√2·sinθ` (per the `‖P_Y−P_Z‖_F` normalization),
    and characterize the pointwise gap `g(θ) = θ − sinθ` (and its `√2`-scaled / `2sin(θ/2)`
    counterparts). Prove `g` is non-negative, strictly increasing on `(0, π/2)`, with
    `g(θ) = θ³/6 + O(θ⁵)` as `θ→0` and maximal endpoint gap at `θ = π/2`
    (`θ = π/2 ≈ 1.5708` vs. `sin θ = 1`). Aggregate over the `p` angles to give the exact
    multivariate gap `d_geo − d_proj` as a function of `(θ_1,…,θ_p)`, and show it is
    consistent in magnitude with an observed ≈`0.64` when one angle is near `π/2`.
  - Prove that `d_geo = √(Σ θ_i²)` is the genuine Riemannian (geodesic, length-minimizing)
    distance for the canonical metric of §1.1 — e.g. by exhibiting the minimizing geodesic
    and showing no shorter admissible path exists — and that every member of the
    projector/chordal family is an *extrinsic chord length*, hence a lower bound
    `d_proj ≤ d_geo` that is a metric on `Gr(p,n)` but **not** the geodesic distance. State
    precisely why the two necessarily diverge as `θ_max → π/2` (arc vs. chord; the chord
    saturates at the diameter of the projector sphere while the arc keeps growing).

**(1b) Cut-locus structure.** The cut locus of `𝒮` consists of subspaces `𝒯` reaching at
least one principal angle `θ_i = π/2` (`σ_i = 0`). Characterize precisely:

  - Why the minimizing geodesic from `𝒮` to such `𝒯` is **non-unique** and the Riemannian
    logarithm (the initial tangent `H` with `exp_𝒮(H) = 𝒯`, `‖H‖ = d_geo`) is
    **multivalued** there. Give the dimension/parametrization of the set of minimizers
    (the choice of principal-vector pairing in the degenerate `σ_i = 0` block, i.e. the
    `O(m)` freedom when `m` angles equal `π/2`).
  - State the correct distance at the cut locus (it remains `√(Σ θ_i²)` with the offending
    `θ_i = π/2`, single-valued even though the log is not) and give a well-defined
    *selection* of one valid logarithm there, proving its norm equals `d_geo`.

**(1c) Optimal conditioning of recovering `θ` from `σ`.** Recovering `θ_i = arccos(σ_i)`
is ill-conditioned near `θ = 0` (`σ = 1`): `d/dσ arccos(σ) = −1/√(1−σ²) → −∞` as `σ → 1`.

  - Derive the exact first-order error amplification: for a perturbation `δσ` of the
    singular value, `δθ ≈ −δσ/√(1−σ²) = −δσ/ sinθ`. Conclude that a 1-ulp error in `σ`
    near `σ = 1` (so `δσ ≲ u`) inflates to `δθ ≈ u / sinθ`, and using `sinθ ≈ √(2(1−σ))`
    show the bound degrades to `δθ ≈ √(2u)` at the worst small-angle scale
    (`≈ √(2·1.11·10^{−16}) ≈ 1.5·10^{−8}`), losing ~half the significant digits — exactly
    the ≈`5.97·10^{−9}`-vs-`10^{−9}` failure scale observed when angles are extracted by
    `arccos(σ)` at the smallest angles.
  - Give and **prove** an optimally-conditioned formula for `θ_i` that attains machine
    precision *uniformly* over `θ ∈ [0, π/2]`. Two routes to analyze:
    (i) the **atan2 of singular values of an off-diagonal block** — split an orthonormal
    completion so that for the pair of frames the principal angles satisfy
    `θ_i = atan2(σ_i^{⟂}, σ_i^{∥})` where `σ^{∥}` are the singular values of `YᵀZ`
    (`=cosθ`) and `σ^{⟂}` are the singular values of `(I − YYᵀ)Z` (`=sinθ`); equivalently
    `θ_i = atan( σ_i(M) )` where `M = (I−YYᵀ)Z·(YᵀZ)⁻¹` has singular values `tanθ_i`
    (so the eigenvalues of `MᵀM` are `tan²θ_i` and `θ_i = arctan(√eval_i)`).
    (ii) the **half-chord arcsine** `θ_i = 2·arcsin( ½‖principal-vector chord‖ )`.
    Prove route (i): show `θ ↦ atan(tanθ)` and `θ ↦ arctan(√(tan²θ))` are well-conditioned
    for all `θ ∈ [0, π/2)` (derivative of `arctan(t)` is `1/(1+t²)`, bounded by 1; the
    composite condition number is `O(1)` away from `π/2`) and that the sine branch handles
    the `θ→π/2` end, so that combining a `cos`-stable branch near `0` and a `sin`-stable
    branch near `π/2` (or the single `atan2(sin,cos)` form) yields uniform relative
    accuracy `O(u)` in `θ` across the whole range, in contrast to `arccos(σ)`. Identify the
    crossover and quantify the worst-case error of each branch.

---

## Problem 2 — Matérn smoothness-dependent penalty enumeration and the κ-derivative index invariant

### 2.1 The Matérn RKHS and its Sobolev order

Fix a spatial dimension `d ≥ 1` and a smoothness parameter `ν > 0`. The **Matérn
covariance kernel** on `ℝ^d` with smoothness `ν`, marginal variance 1, and inverse
length-scale `κ = e^ψ > 0` (we work in the log-parameter `ψ = log κ`) is
```
    k_ν(r) = (2^{1−ν}/Γ(ν)) (κ r)^ν K_ν(κ r),   r = ‖x − x'‖,
```
where `K_ν` is the modified Bessel function of the second kind. Its reproducing-kernel
Hilbert space (RKHS) `H_ν` is norm-equivalent to the Sobolev space `H^m(ℝ^d)` with
**Sobolev order**
```
    m = ν + d/2.
```
This is the classical spectral fact: the Matérn spectral density decays as
`(κ² + ‖ω‖²)^{−(ν + d/2)}`, so the RKHS squared norm `‖f‖²_{H_ν} ≍ ∫ |f̂(ω)|² (κ² + ‖ω‖²)^m dω`
controls exactly the derivatives of `f` up to order `m`: the order-`j` derivative seminorm
`‖D^j f‖_{L²}² = ∫ ‖ω‖^{2j} |f̂(ω)|² dω` is finite (controlled by the RKHS norm) precisely
when `j ≤ m`, and a derivative-`j` penalty with `j > m` imposes roughness control the
kernel's own RKHS does **not** possess.

### 2.2 The operator-penalty overlay and its smoothness gate

A reduced-rank Matérn smooth is fit with an overlay of differential-operator penalties
built from collocation matrices `D_0, D_1, D_2` (discretizations of the value `D^0=I`,
gradient `D^1=∇`, and Hessian `D^2=∇²` operators on the basis), giving squared-`L²`-seminorm
penalties
```
    S_j = c_j^{−1} · D_jᵀ D_j     (j = 0: mass; j = 1: tension; j = 2: stiffness),
```
each normalized by its own Frobenius scale `c_j = ‖D_jᵀD_j‖_F`. The **smoothness gate**
admits operator `j` into the penalty set iff its seminorm is finite on the Matérn RKHS,
i.e. iff `j < m` *strictly* (with a small tolerance so an exact half-integer boundary
disables the matching-order operator):
```
    admissible(j) ⇔ m > j   ⇔ ν + d/2 > j.
```
Concretely: mass (`j=0`) is always admitted; tension (`j=1`) requires `m > 1`; stiffness
(`j=2`) requires `m > 2`. For `ν ≥ 3/2`, or for any `d ≥ 2`, all three are admitted; only
the genuinely rough `ν = 1/2, d = 1` case (`m = 1`, the Ornstein–Uhlenbeck / exponential
kernel, an `H¹` process with continuous but non-differentiable paths) drops tension and
stiffness, leaving the single mass penalty. Admitting `j = 1` there would bias the
reduced-rank fit toward `C¹` functions the kernel does not favour, over-smoothing relative
to the exact GP and collapsing held-out oscillation.

The **forward** penalty builder emits the gated list `{ S_j : j < m }` — its length is
`G(ν,d) = #{ j ∈ {0,1,2} : ν + d/2 > j }` (so `G = 1` for `ν=1/2,d=1`, `G = 3` for the
admit-all cases). For smoothness optimization, gradients are taken with respect to
`ψ = log κ`; a separate builder produces the **ψ-derivatives** `∂S_j/∂ψ` (and second
derivatives `∂²S_j/∂ψ²`) of those same penalties via the exact quotient/chain rule on the
normalized Grams.

### 2.3 The defect (the structure to repair, stated as a mathematical inconsistency)

The forward builder enumerates the **gated** set `{ S_j : j < m }` of length `G(ν,d)`. The
ψ-derivative builder instead enumerates the **ungated** triple `[∂S_0/∂ψ, ∂S_1/∂ψ,
∂S_2/∂ψ]` of fixed length 3, regardless of `(ν,d)`. When `G(ν,d) < 3` — i.e. for a rough
Matérn (`m ≤ 2`, in particular `ν=1/2,d=1` with `m=1` and a single, non-double penalty) —
the two lists are **index-misaligned**: the optimizer pairs each penalty `S_{a}` (indexed
in the gated forward list) with smoothing weight `λ_a` but reads the derivative entry at
the same position `a` from the ungated list, which corresponds to a *different* operator
`j`. The assembled `ψ`-gradient of the penalized criterion is then taken against a
mismatched penalty, so it is not the derivative of the objective actually being optimized.

### 2.4 Asks

**(2a) The correct `(ν,d) → admissible-penalty-set` rule from Matérn RKHS theory.** State
and prove, from the spectral characterization of the Matérn RKHS as `H^{ν+d/2}(ℝ^d)`,
exactly which derivative-operator seminorms `‖D^j f‖_{L²}` are finite (a.s. / RKHS-bounded)
for given `(ν, d)`. Precisely:

  - Prove that `‖D^j f‖_{L²}² = ∫ ‖ω‖^{2j} |f̂(ω)|² dω` is bounded by a constant times
    `‖f‖²_{H_ν}` iff `‖ω‖^{2j} / (κ² + ‖ω‖²)^{ν+d/2}` is bounded as `‖ω‖→∞`, i.e. iff
    `2j ≤ 2(ν + d/2)`, i.e. `j ≤ ν + d/2 = m`. Distinguish the boundary case `j = m`
    (the seminorm is exactly the borderline-divergent / non-controlled case; explain why
    the strict gate `j < m` is the correct admissibility rule for a *finite, kernel-honest*
    penalty and not merely a numerical convenience), and confirm the resulting count
    `G(ν,d) = #{ j ∈ {0,1,2} : j < ν + d/2 }`. Discuss the `ν=1/2, d=1` exponential-kernel
    case as the canonical `m=1` boundary where only `j=0` survives.

**(2b) The index-alignment invariant.** Prove the consistency requirement that the
`κ`-derivative penalty list must enumerate the **same gated set** as the forward penalty
list, in the **same order**. Formalize as follows. Let the penalized objective be
`F(β, ψ) = ℓ(β) + ½ Σ_{a=1}^{G} λ_a · βᵀ S_{(a)}(ψ) β`, where `(a) ↦ j` is the gated
indexing map (the `a`-th admitted operator order) and each `S_{(a)}(ψ)` depends on `ψ`
through `κ = e^ψ`. The exact `ψ`-gradient of the penalty block is
`∂F/∂ψ = ½ Σ_{a=1}^{G} λ_a · βᵀ (∂S_{(a)}/∂ψ) β`. Prove:

  - (Correctness ⇔ alignment) The assembled gradient equals the true `∂F/∂ψ` **iff** the
    derivative list supplied to position `a` is `∂S_{(a)}/∂ψ` for the *same* gated index
    map `(a)` used by the forward list — i.e. iff the derivative builder applies the
    identical gate `j < m`. If the derivative list is the ungated triple while the forward
    list is gated, positions `a ≥ 2` reference `∂S_{j'}/∂ψ` for `j' ≠ (a)`, and the
    resulting vector is the gradient of a *different* functional (a wrong-operator penalty),
    not of `F`; quantify the error as `½ Σ_a λ_a βᵀ (∂S_{j'(a)}/∂ψ − ∂S_{(a)}/∂ψ) β`, which
    is generically nonzero whenever any admitted operator differs in the two enumerations.
  - (Why it is latent under double-penalty configs) Note that if the configuration always
    admits all three operators (e.g. the `m > 2` regime, or a "double-penalty" mode that
    forces the full triple), then `G = 3`, the gate is vacuous, both lists coincide, and
    the desync cannot manifest — establishing that the inconsistency is exposed *only* in a
    rough-`ν`, non-double-penalty regime (`G < 3`). State the general invariant: *any pair
    of (penalty list, penalty-derivative list) consumed positionally by a gradient
    assembler must be generated by one and the same admissibility predicate and ordering*,
    and prove this is necessary and sufficient for the assembled `ψ`-gradient to equal the
    analytic derivative of the penalized objective for every `(ν, d)`.


---

## What we want back, per problem

For each numbered problem: (1) the exact object or identity, derived; (2) a proof of correctness — including a proof that it reduces to the stated known special case and extends to the failing regime; (3) where relevant, an **independent self-checking identity** (a scalar or matrix equality that two distinct derivations must satisfy) that could serve as a correctness certificate without running code; (4) a clear statement of any regime where the problem is genuinely ill-posed, with the impossibility argument. Pseudocode is welcome as mathematics; source code in any language is neither needed nor wanted.

Prefer one fully-correct general theorem to several special cases. If you see a deeper unifying structure across the three sections (they are all, in different guises, about differentiating a constrained/profiled criterion through a moving subspace), say so and exploit it.
