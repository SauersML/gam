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
