# Mathematical Questions for Unifying the Outer Optimizer

## Background

We have a penalized regression framework (generalized additive models) with a two-tier optimization:

- **Inner problem**: Find coefficients $\hat\beta(\theta)$ minimizing the penalized negative log-likelihood:
  $$F(\beta, \theta) = -\ell(\beta) + \tfrac{1}{2}\beta^\top S(\theta)\,\beta$$
  The stationarity condition is $\nabla_\beta F = 0$, giving the penalized Hessian $H(\theta) = -\nabla^2_\beta \ell + S(\theta)$.

- **Outer problem**: Select hyperparameters $\theta$ by minimizing the REML/LAML (restricted/Laplace approximate marginal likelihood) objective:
  $$V(\theta) \;=\; F\bigl(\hat\beta(\theta),\theta\bigr) + \tfrac{1}{2}\log|H(\theta)|_+ - \tfrac{1}{2}\log|S(\theta)|_+ + \text{const}$$
  where $|\cdot|_+$ denotes the product of positive eigenvalues.

### Current state

We have a single "unified evaluator" function that, given $\hat\beta$, $H$, penalty square roots $R_k$, and a **derivative provider** (explained below), computes $V$, $\nabla_\theta V$, and $\nabla^2_\theta V$.

**This evaluator currently handles only $\theta = \rho$**, where $\rho_k = \log\lambda_k$ are log-smoothing-parameters. Under this parameterization:
- $S(\rho) = \sum_k e^{\rho_k} S_k$ where $S_k$ are fixed penalty matrices
- $\frac{\partial S}{\partial \rho_k} = \lambda_k S_k =: A_k$ (the penalty derivative)
- The design matrix $X$ does not depend on $\rho$

**The problem**: We also have hyperparameters $\psi$ (called "directional hyperparameters") that move the design matrix and penalty structure. These are currently handled by a separate 650-line function that reimplements the entire profiled calculus from scratch. We want to fold $\psi$ into the unified evaluator.

---

## 1. How $\psi$ Enters the Problem

### 1.1 Design perturbation

When $\psi$ varies, the design matrix moves:
$$X(\psi) = X_0 + \sum_j \psi_j\, X_{\tau_j} + \tfrac{1}{2}\sum_{i,j}\psi_i\psi_j\, X_{ij}$$

where $X_{\tau_j} = \frac{\partial X}{\partial \psi_j}\big|_{\psi=0}$ is $n \times p$ and $X_{ij} = \frac{\partial^2 X}{\partial \psi_i \partial \psi_j}\big|_{\psi=0}$ is also $n \times p$.

**Concrete example**: For a thin-plate spline in 2D, $\psi$ might control the anisotropy ratio (how much to stretch the x-axis vs y-axis). As $\psi$ moves, the basis functions change, so $X$ changes.

### 1.2 Penalty perturbation

Similarly:
$$S(\rho, \psi) = \sum_k \lambda_k \Bigl[S_k^{(0)} + \sum_j \psi_j\, S_{k,\tau_j} + \tfrac{1}{2}\sum_{i,j}\psi_i\psi_j\, S_{k,ij}\Bigr]$$

So $S$ depends on both $\rho$ (through $\lambda_k = e^{\rho_k}$) and $\psi$ (through the penalty structure).

### 1.3 Likelihood Hessian perturbation

Because $X$ moves with $\psi$, the likelihood $\ell(\beta; X(\psi))$ and its Hessian $H_L(\beta, \psi) = -\nabla^2_\beta \ell$ also depend on $\psi$ directly (not just through $\beta$).

---

## 2. The Profiled Calculus for $\theta = (\rho, \psi)$

### 2.1 Fixed-beta objects

For each hyperparameter coordinate $\theta_i$ (which can be either a $\rho_k$ or a $\psi_j$), define:

$$v_i = \frac{\partial V}{\partial \theta_i}\bigg|_{\beta\text{ fixed}}, \quad
g_i = \frac{\partial (\nabla_\beta F)}{\partial \theta_i}\bigg|_{\beta\text{ fixed}}, \quad
\dot{H}_i^{(\text{fix})} = \frac{\partial H}{\partial \theta_i}\bigg|_{\beta\text{ fixed}}$$

**For $\rho_k$**: $v_k = \frac{1}{2}\hat\beta^\top A_k \hat\beta$, $\;g_k = A_k\hat\beta$, $\;\dot{H}_k^{(\text{fix})} = A_k$.

**For $\psi_j$**: The family must provide $\frac{\partial(-\ell)}{\partial\psi_j}\big|_\beta$, $\;\frac{\partial\nabla_\beta(-\ell)}{\partial\psi_j}\big|_\beta$, and $\;\frac{\partial H_L}{\partial\psi_j}\big|_\beta$. The penalty adds $\frac{\partial S}{\partial\psi_j}\hat\beta$ to $g_j$ and $\frac{\partial S}{\partial\psi_j}$ to $\dot{H}_j^{(\text{fix})}$.

### 2.2 Mode response (first-order)

By implicit differentiation of $\nabla_\beta F(\hat\beta(\theta), \theta) = 0$:
$$\beta_i := \frac{\partial\hat\beta}{\partial\theta_i} = -H^{-1}g_i$$

### 2.3 Total Hessian drift (first-order)

The total first derivative of $H$ with respect to $\theta_i$ (including the implicit $\beta$-movement) is:
$$\dot{H}_i = \dot{H}_i^{(\text{fix})} + D_\beta H_L[\beta_i]$$

where $D_\beta H_L[u]$ is the directional derivative of the likelihood Hessian along direction $u$:
$$D_\beta H_L[u] = \lim_{\epsilon\to 0}\frac{H_L(\beta+\epsilon u) - H_L(\beta)}{\epsilon}$$

### 2.4 Profiled gradient

$$\frac{\partial V}{\partial\theta_i} = v_i + \tfrac{1}{2}\operatorname{tr}(H^{-1}\dot{H}_i) - \tfrac{1}{2}\frac{\partial\log|S|_+}{\partial\theta_i}$$

(The $g_i^\top\beta_i$ cross-term vanishes by the envelope theorem.)

### 2.5 Mode response (second-order)

$$\beta_{ij} = -H^{-1}\bigl(g_{ij} + \dot{H}_i^{(\text{fix})}\beta_j + \dot{H}_j^{(\text{fix})}\beta_i + D_\beta H_L[\beta_i]\,\beta_j\bigr)$$

where $g_{ij} = \frac{\partial^2(\nabla_\beta F)}{\partial\theta_i\partial\theta_j}\big|_\beta$.

### 2.6 Total Hessian drift (second-order)

$$\ddot{H}_{ij} = \dot{H}_{ij}^{(\text{fix})} + D_\beta H_L[\beta_{ij}] + D_\beta\dot{H}_i^{(\text{fix})}[\beta_j] + D_\beta\dot{H}_j^{(\text{fix})}[\beta_i] + D^2_\beta H_L[\beta_i, \beta_j]$$

### 2.7 Profiled Hessian

$$\frac{\partial^2 V}{\partial\theta_i\partial\theta_j} = \bigl(v_{ij} - g_i^\top H^{-1}g_j\bigr) + \tfrac{1}{2}\bigl[\operatorname{tr}(H^{-1}\ddot{H}_{ij}) - \operatorname{tr}(H^{-1}\dot{H}_j\,H^{-1}\dot{H}_i)\bigr] - \tfrac{1}{2}\frac{\partial^2\log|S|_+}{\partial\theta_i\partial\theta_j}$$

---

## 3. What the Unified Evaluator Currently Does

The evaluator receives an `InnerSolution` containing: $\hat\beta$, $H$ (via a factorized operator), penalty roots $R_k$ (so $S_k = R_k^\top R_k$), log-determinant derivatives of $S$, and a **derivative provider**.

The derivative provider is a trait with two methods:
- `correction(v_k)` $\to D_\beta H_L[-v_k]$ where $v_k = H^{-1}(A_k\hat\beta)$
- `second_correction(v_k, v_l, u_{kl})` $\to D_\beta H_L[u_{kl}] + D^2_\beta H_L[-v_l, -v_k]$

The evaluator then computes the gradient loop ($k = 1,\ldots,K$):
1. $A_k = \lambda_k R_k^\top R_k$
2. $v_k = H^{-1}(A_k\hat\beta)$
3. $\dot{H}_k = A_k + \text{correction}(v_k)$
4. Gradient entry $= \frac{1}{2}\hat\beta^\top A_k\hat\beta + \frac{1}{2}\operatorname{tr}(H^{-1}\dot{H}_k) - \frac{1}{2}(\log|S|_+)'_k$

And the Hessian loop (for all $k,l$ pairs) using the second correction.

---

## 4. Questions for the Math Team

### Q1: Can the $\psi$ gradient be expressed through the same provider interface?

The evaluator's gradient loop is:
$$\frac{\partial V}{\partial\rho_k} = v_k + \tfrac{1}{2}\operatorname{tr}\bigl(H^{-1}[A_k + \text{correction}(v_k)]\bigr) - \tfrac{1}{2}(\log|S|_+)'_k$$

For a $\psi_j$ direction, the analogous formula is:
$$\frac{\partial V}{\partial\psi_j} = v_j + \tfrac{1}{2}\operatorname{tr}\bigl(H^{-1}[\dot{H}_j^{(\text{fix})} + D_\beta H_L[\beta_j]]\bigr) - \tfrac{1}{2}(\log|S|_+)'_j$$

**The difference**: For $\rho_k$, the fixed-beta Hessian drift is simply $A_k = \lambda_k S_k$ (known from penalty structure). For $\psi_j$, the fixed-beta Hessian drift $\dot{H}_j^{(\text{fix})}$ involves the family-specific $\frac{\partial H_L}{\partial\psi_j}\big|_\beta$ plus $\frac{\partial S}{\partial\psi_j}$.

**Question**: Can we generalize the evaluator to handle both $\rho$ and $\psi$ by replacing the penalty-root-based $A_k$ with a more general "fixed-beta Hessian drift" object? Specifically, instead of the loop building $A_k$ from roots, we'd supply for each coordinate $i$:
- The fixed-beta cost derivative $v_i$
- The fixed-beta score $g_i$ (for mode response and Hessian)
- The fixed-beta Hessian drift $\dot{H}_i^{(\text{fix})}$ (for traces)
- The log-det-S derivative $(\log|S|_+)'_i$

And the evaluator does the rest (mode response, correction, trace, assembly). Is this mathematically sound? Are there terms that break this factorization?

### Q2: The "relinearization" identity

At current $\psi$, the effective first-order design derivative is:
$$\tilde{X}_{\tau_j} = X_{\tau_j} + \sum_i \psi_i\,X_{ij}$$

This arises because $\frac{\partial X(\psi)}{\partial\psi_j} = X_{\tau_j} + \sum_i \psi_i X_{ij}$.

**Question**: When computing the profiled gradient at a general $\psi \neq 0$, should the evaluator work with the **relinearized** design derivatives $\tilde{X}_{\tau_j}$ (absorbing the current $\psi$), or with the original $X_{\tau_j}$? Does the relinearization affect the penalty log-determinant terms or the mode response?

### Q3: Cross-derivatives $\frac{\partial^2 V}{\partial\rho_k\partial\psi_j}$

The mixed second derivative involves:
- $g_{k,j}$: How does $\nabla_\beta F$ change when both $\rho_k$ and $\psi_j$ move?
- $\dot{H}_k^{(\text{fix})}$ applied to $\beta_j$ and vice versa
- The second mode response $\beta_{kj}$

**For pure $\rho$-$\rho$**: $g_{kl} = 0$ when $k \neq l$ and $g_{kk} = A_k\hat\beta$ (from $\frac{\partial^2}{\partial\rho_k^2}[\lambda_k S_k\hat\beta] = \lambda_k S_k\hat\beta$).

**For $\rho_k$-$\psi_j$ cross**: $g_{k,j} = \frac{\partial(A_k\hat\beta)}{\partial\psi_j}\big|_\beta = \lambda_k\frac{\partial S_k}{\partial\psi_j}\hat\beta$ if $S_k$ depends on $\psi$, and also $\frac{\partial g_j}{\partial\rho_k}$.

**Question**: Please write out the complete formula for $\frac{\partial^2 V}{\partial\rho_k\partial\psi_j}$ explicitly, identifying which terms are "generic" (penalty-structure-only) and which require family-specific callbacks. Can the cross-derivative be expressed in terms of the same correction/second_correction interface, or does it need new mathematical objects?

### Q4: The $D_\beta\dot{H}_i^{(\text{fix})}[\beta_j]$ term

In the second-order Hessian drift $\ddot{H}_{ij}$, there's a term $D_\beta\dot{H}_i^{(\text{fix})}[\beta_j]$. For pure $\rho$ coordinates, $\dot{H}_k^{(\text{fix})} = A_k$ is independent of $\beta$, so this term vanishes.

**For $\psi$ coordinates**, $\dot{H}_j^{(\text{fix})}$ includes $\frac{\partial H_L}{\partial\psi_j}\big|_\beta$ which itself depends on $\beta$. So:
$$D_\beta\dot{H}_j^{(\text{fix})}[\beta_i] = D_\beta\Bigl(\frac{\partial H_L}{\partial\psi_j}\bigg|_\beta\Bigr)[\beta_i]$$

This is a **third-order mixed derivative** of the log-likelihood: $\frac{\partial^3(-\ell)}{\partial\beta^2\partial\psi_j}$ contracted with direction $\beta_i$.

**Question**:
(a) Under what conditions does this term vanish or simplify? (e.g., canonical exponential families, Gaussian, logistic)
(b) For the link-wiggle model where $\eta = g(X\beta; \theta)$ with $g$ being the wiggled link, can this be expressed as a function of the existing weight derivatives $W'$ and $W''$?
(c) Is this term needed for practical convergence of the outer optimizer, or can it be safely dropped (set to zero) without affecting the quality of the Newton step?

### Q5: What fixed-beta objects must the family provide for $\psi$?

Currently, for $\rho$ directions, the evaluator needs nothing from the family beyond the inner solution — all fixed-beta objects ($v_k$, $g_k$, $A_k$) come from the penalty structure.

**For $\psi$ directions**, the family must provide (per $\psi_j$):
1. $\frac{\partial(-\ell)}{\partial\psi_j}\big|_\beta$ (scalar)
2. $\frac{\partial\nabla_\beta(-\ell)}{\partial\psi_j}\big|_\beta$ (p-vector)
3. $\frac{\partial H_L}{\partial\psi_j}\big|_\beta$ (p$\times$p matrix)

And for second-order (per pair $\psi_i, \psi_j$):
4. $\frac{\partial^2(-\ell)}{\partial\psi_i\partial\psi_j}\big|_\beta$ (scalar)
5. $\frac{\partial^2\nabla_\beta(-\ell)}{\partial\psi_i\partial\psi_j}\big|_\beta$ (p-vector)
6. $\frac{\partial^2 H_L}{\partial\psi_i\partial\psi_j}\big|_\beta$ (p$\times$p matrix)

**Question**: Is this the minimal and complete set? Are there additional objects needed for the cross $\rho$-$\psi$ terms beyond what's listed above plus the penalty structure derivatives $\frac{\partial S_k}{\partial\psi_j}$?

### Q6: Profiled dispersion under $\psi$

For Gaussian models, the dispersion $\phi$ is profiled:
$$\hat\phi(\theta) = \frac{D_p(\hat\beta(\theta), \theta)}{n - M_p}$$

where $D_p = -2\ell + \hat\beta^\top S\hat\beta$ is the penalized deviance.

When $\psi$ changes, $D_p$ changes both through $\hat\beta(\psi)$ and through $X(\psi)$ (changing $\ell$) and $S(\psi)$ (changing the penalty). The REML objective for Gaussian with profiled $\phi$ is:
$$V(\theta) = \frac{n - M_p}{2}\log\hat\phi + \frac{1}{2}\log|H|_+ - \frac{1}{2}\log|S|_+ + \text{const}$$

**Question**: Write out $\frac{\partial V}{\partial\psi_j}$ for the profiled-Gaussian case. Does the chain rule through $\hat\phi(\theta)$ introduce additional terms beyond the general LAML formula, or does it simplify?

---

## 5. Separate Question: Sharing Likelihood Between REML and HMC

We also have an HMC (Hamiltonian Monte Carlo) posterior sampler that currently hardcodes two likelihood functions:

**Bernoulli-logit**:
$$\log p(y|\eta) = \sum_i w_i\bigl[y_i\eta_i - \log(1+e^{\eta_i})\bigr], \quad \eta = X\beta$$

**Gaussian-identity**:
$$\log p(y|\eta) = -\frac{1}{2}\sum_i w_i(y_i - \eta_i)^2$$

The HMC target is the penalized conditional posterior:
$$\log p(\beta|y,\lambda) = \ell(y|\beta) - \tfrac{1}{2}\beta^\top S(\lambda)\beta$$

with $\lambda$ fixed from the REML fit. The sampler uses a whitening transform $L$ from $H^{-1} = LL^\top$ to reparameterize: $\beta = \hat\beta + Lz$, then runs NUTS in $z$-space.

The REML evaluator uses the same $\ell(y|\beta)$ but adds the Laplace correction $\frac{1}{2}\log|H|$. Both need $\ell$ and $\nabla_\beta\ell$.

**Question**: What is the cleanest mathematical factorization that lets both REML and HMC share the likelihood? We want:
- REML to evaluate $V(\theta)$, $\nabla_\theta V$, $\nabla^2_\theta V$ (needs $\ell$, $H_L$, $D_\beta H_L$, etc.)
- HMC to evaluate $\log p(\beta|y,\lambda)$ and $\nabla_\beta\log p$ very fast (millions of evaluations per chain, only needs $\ell$ and $\nabla_\beta\ell$)

Is the right abstraction a "likelihood oracle" providing $(\ell, \nabla_\beta\ell)$ and optionally $(H_L, D_\beta H_L, D^2_\beta H_L)$ on demand? Or is there a better structure that avoids overhead for HMC while giving REML everything it needs?

---

## 6. Link-Wiggle Specific Question

For link-wiggle models, the linear predictor is:
$$\eta = g(X\beta;\, \theta), \quad g(u;\theta) = u + B(u)\theta$$

where $B(u)$ is a B-spline basis evaluated at the base predictor $u = X\beta$, and $\theta$ are the wiggle coefficients (jointly estimated with $\beta$).

The joint Jacobian is:
$$J = \begin{pmatrix} \text{diag}(g'(u))\,X & B(u) \end{pmatrix}$$

and the Gauss-Newton Hessian is $H_L = J^\top W J$.

The first directional derivative $D_\beta H_L[\delta]$ has three terms:
1. **Jacobian sensitivity**: $\dot{J}^\top WJ + J^\top W\dot{J}$, where $\dot{J}$ involves $g''(u)\cdot X\delta_\beta$ and $B'(u)\cdot\frac{du}{d\text{something}}\cdot\delta_\theta$
2. **Weight sensitivity**: $J^\top\text{diag}(W'\cdot J\delta)\,J$
3. Combined: $D_\beta H_L[\delta] = \text{(Jacobian pair)} + \text{(weight correction)}$

**Question**: For link wiggles, the "hyperparameters" $\theta$ are actually part of the coefficient vector (they're jointly optimized in the inner problem). But we also want to penalize them with their own smoothing parameter $\lambda_\theta$. This creates a block structure:
$$\alpha = \begin{pmatrix}\beta \\ \theta\end{pmatrix}, \quad S = \begin{pmatrix}\sum_k\lambda_k S_k & 0 \\ 0 & \lambda_\theta S_\theta\end{pmatrix}$$

The REML optimization is over $\rho = (\log\lambda_1, \ldots, \log\lambda_K, \log\lambda_\theta)$. There is no separate $\psi$ for link wiggles — the wiggle coefficients are inner variables, not outer hyperparameters.

**However**, for spatial smooths (e.g., thin-plate splines with anisotropy), the $\psi$ directions move the basis, which is mathematically equivalent to a link wiggle with $\theta$ promoted to an outer hyperparameter.

**Question**: Is there a unified framework where:
- Link wiggles treat $\theta$ as **inner** (optimized jointly with $\beta$, penalized by $\lambda_\theta$)
- Spatial anisotropy treats $\psi$ as **outer** (optimized in the outer loop via profiled REML)
- Both use the same directional derivative machinery for $D_\beta H_L$?

What mathematical conditions distinguish "inner" from "outer" hyperparameters?

---

## 7. Additional Questions (from external review)

### Q7: Firth bias reduction in multi-block / joint space

Firth bias reduction adds a penalty $\frac{1}{2}\log|I(\beta)|$ to the log-likelihood, where $I(\beta)$ is the Fisher information matrix. For a single-predictor GLM, $I = X^\top W X$ and the Firth-corrected score is well-known (Firth 1993).

**For multi-block models** (e.g., GAMLSS with location $\mu$ and scale $\sigma$ jointly estimated), the Fisher information is the **full joint** information matrix coupling all predictors:
$$I(\alpha) = J^\top W J$$
where $\alpha = (\beta_\mu, \beta_\sigma, \ldots)$ and $J$ is the joint Jacobian.

The Firth-corrected REML objective would be:
$$V_{\text{Firth}}(\theta) = V(\theta) + \tfrac{1}{2}\log|I(\hat\alpha(\theta))|$$

**Questions**:
(a) What is $\frac{\partial}{\partial\rho_k}\bigl[\tfrac{1}{2}\log|I(\hat\alpha(\rho))|\bigr]$? The chain rule through $\hat\alpha(\rho)$ introduces the mode response, so this involves $D_\alpha\log|I|[\alpha_k]$ where $\alpha_k = -H^{-1}g_k$.

(b) For block-diagonal $I$ (uncoupled predictors), does the Firth gradient decompose block-locally? i.e., can each block compute its own Firth correction independently?

(c) For coupled blocks (e.g., location-scale with shared observations), how does the cross-block coupling in $I$ affect the Firth gradient? Is there a tractable formula, or does one need a full $p_{\text{total}} \times p_{\text{total}}$ sensitivity solve?

### Q8: Smoothness of the pseudo-determinant $\log|S(\theta)|_+$

The REML objective uses $\log|S(\theta)|_+$, the log-product of positive eigenvalues of $S(\theta)$. The rank of $S$ is assumed structurally fixed (determined by the null space of the penalty, e.g., polynomials for spline penalties).

**Problem**: As $\theta$ varies (especially $\psi$ directions that change penalty structure), eigenvalues of $S$ can cross zero, causing:
- $\log|S|_+$ to have a discontinuous gradient (the rank changes)
- Newton/ARC steps to become unreliable near the rank-transition boundary

**Questions**:
(a) Under what conditions on $S(\theta)$ is $\log|S(\theta)|_+$ smooth (i.e., $C^2$) over the entire optimization domain? Is it sufficient that each $S_k$ has a fixed null space?

(b) When smoothness fails, what is the correct remedy? Options:
  - **Soft rank**: Replace $\log|S|_+ = \sum_{\sigma_i > 0}\log\sigma_i$ with a smooth approximation like $\sum_i \log(\sigma_i + \epsilon)$ or $\sum_i \log\max(\sigma_i, \epsilon)$. Does this bias the REML estimate?
  - **Analytic continuation**: Is there a formula for $\log|S|_+$ that remains analytic even through rank transitions?
  - **Constraint**: Should the optimizer be constrained to the manifold where $\text{rank}(S)$ is constant?

(c) For the specific case $S(\rho,\psi) = \sum_k \lambda_k S_k(\psi)$, if each $S_k(\psi)$ has constant rank but the combined $S$ does not, does this cause problems in practice?

### Q9: Cross-predictor curvature in location-scale models

In a GAMLSS (location-scale) model with two predictors $\eta_\mu = X_\mu\beta_\mu$ and $\eta_\sigma = X_\sigma\beta_\sigma$, the joint negative log-likelihood Hessian has the block structure:
$$H_L = \begin{pmatrix} H_{\mu\mu} & H_{\mu\sigma} \\ H_{\sigma\mu} & H_{\sigma\sigma} \end{pmatrix}$$

where the off-diagonal blocks $H_{\mu\sigma}$ represent the curvature coupling between location and scale.

For a Gaussian location-scale model with $y_i \sim N(\mu_i, \sigma_i^2)$:
$$H_{\mu\mu} = X_\mu^\top\text{diag}(1/\sigma_i^2)\,X_\mu, \quad H_{\mu\sigma} = X_\mu^\top\text{diag}(2r_i/\sigma_i^2)\,X_\sigma$$
where $r_i = (y_i - \mu_i)/\sigma_i$.

**Questions**:
(a) The directional derivative $D_\beta H_L[u]$ for the joint system must account for how perturbing $\beta_\mu$ affects $H_{\sigma\sigma}$ (through $r_i$) and vice versa. Write out $D_\beta H_L[u]$ explicitly for the Gaussian location-scale case, showing all cross-block terms.

(b) Can the $D_\beta H_L[u]$ for coupled blocks be decomposed as a sum of "within-block" terms (expressible as $X_b^\top\text{diag}(\cdot)X_b$) and "cross-block" terms (expressible as $X_a^\top\text{diag}(\cdot)X_b$)? If so, what are the weight vectors for each term?

(c) For the second directional derivative $D^2_\beta H_L[u, v]$, how many independent "weight-like" vectors are needed? The single-predictor case needs $W'$ and $W''$. The coupled case presumably needs partial derivatives of each block of $H_L$ w.r.t. each linear predictor.

### Q10: Corrected covariance $V_\beta^*$ under $\psi$

Wood (2016) defines the corrected covariance for smoothing parameter uncertainty:
$$V_\beta^* = V_\beta + V_\beta\,\nabla_\beta^2 V \cdot (\nabla_\theta^2 V)^{-1}\cdot \nabla_\beta^2 V\, V_\beta$$

(simplified; the actual formula involves mixed $\beta$-$\theta$ derivatives of the LAML objective).

**Questions**:
(a) When $\theta = (\rho, \psi)$, does the $V_\beta^*$ correction require new mathematical objects beyond those needed for $\nabla^2_\theta V$? Or is it purely a function of the outer Hessian and the $\beta$-$\theta$ coupling that's already computed?

(b) For the psi directions, the $\beta$-$\theta$ coupling involves $\frac{\partial\hat\beta}{\partial\psi_j} = -H^{-1}g_j$ where $g_j$ includes family-specific terms. Does this coupling affect the corrected covariance qualitatively differently from the $\rho$ coupling?

### Q11: Non-polynomial hyperparameters (Matérn $\kappa$, SAS $\epsilon$)

The $\psi$ framework in Section 1.1 assumes a second-order Taylor expansion:
$$X(\psi) = X_0 + \sum_j\psi_j X_{\tau_j} + \tfrac{1}{2}\sum_{i,j}\psi_i\psi_j X_{ij}$$

This is exact when $X$ depends on $\psi$ polynomially (e.g., anisotropy ratios that linearly scale coordinates).

**But** some hyperparameters enter non-polynomially:
- **Matérn length scale $\kappa$**: The Matérn covariance $C(r) = \frac{2^{1-\nu}}{\Gamma(\nu)}(\kappa r)^\nu K_\nu(\kappa r)$ depends on $\kappa$ through a Bessel function. The penalty $S(\kappa) = C(\kappa)^{-1}$ and the basis $X(\kappa)$ (if using a finite-element representation) are non-polynomial in $\kappa$.
- **SAS link shape $\epsilon$**: The sinh-arcsinh link $g(u; \epsilon) = \sinh(\epsilon^{-1}\sinh^{-1}(u) + \delta)$ depends non-polynomially on $\epsilon$.

**Questions**:
(a) Can the profiled calculus from Section 2 be applied to non-polynomial hyperparameters by simply computing $X_\tau = \frac{\partial X}{\partial\kappa}$ and $X_{\tau\tau} = \frac{\partial^2 X}{\partial\kappa^2}$ numerically at the current $\kappa$, and treating the optimization as a sequence of local quadratic models?

(b) If so, is the truncation error (ignoring $O(\Delta\kappa^3)$ terms) acceptable for Newton steps, or does it cause convergence issues? Does the outer optimizer need to be aware that the model is only locally quadratic?

(c) Is there a better approach for non-polynomial hyperparameters — e.g., embedding them as separate coordinates in $\theta$ with their own "hyper-gradient" computed by automatic differentiation, rather than the Taylor expansion framework?

### Q12: Reparameterization invariance

Before fitting, the design matrix $X$ may be column-conditioned: $\tilde{X} = X D^{-1}$ where $D$ is a diagonal scaling matrix (e.g., column standard deviations). The fitted coefficients are $\tilde\beta = D\beta$, and the Hessian transforms as $\tilde{H} = D^{-1}HD^{-1}$.

**Questions**:
(a) Is the REML/LAML objective $V(\theta)$ invariant under this reparameterization? i.e., does $V(\theta; X, S) = V(\theta; \tilde{X}, \tilde{S})$ where $\tilde{S} = D^{-1}SD^{-1}$? If so, prove it. If not, identify which terms break invariance.

(b) The penalty $\log|S|_+$ transforms as $\log|\tilde{S}|_+ = \log|S|_+ - 2\sum_{j\in\text{range}}\log d_j$. Does this constant offset matter for optimization (it shouldn't affect $\nabla_\theta V$), or are there subtle issues with the nullspace dimension $M_p$?

(c) For the profiled Gaussian case, the dispersion $\hat\phi = D_p/(n-M_p)$ involves the penalized deviance $D_p = \|y - X\hat\beta\|^2 + \hat\beta^\top S\hat\beta$. Is $D_p$ invariant under column conditioning? (The residuals $y - X\hat\beta = y - \tilde{X}\tilde\beta$ are invariant, and $\hat\beta^\top S\hat\beta = \tilde\beta^\top\tilde{S}\tilde\beta$, so yes — but confirm.)

### Q13: The dual-inverse paradox for indefinite Hessians

In non-Gaussian, highly flexible models, the penalized likelihood Hessian $H$ can become indefinite (have negative eigenvalues). This happens when the model is flexible enough that the likelihood curvature $H_L$ overwhelms the penalty $S$ in some directions.

The LAML objective needs two things from $H$:
1. **Log-determinant**: $\frac{1}{2}\log|H|_+$ uses only positive eigenvalues (the pseudo-determinant)
2. **Mode response**: $\beta_k = -H^{-1}g_k$ uses the inverse for the implicit function theorem

When $H$ is indefinite, we currently use **two different operators**:
- For the log-determinant and trace terms: the **positive-part pseudo-inverse** $H_+^\dagger = \sum_{\sigma_i > 0} \sigma_i^{-1} u_i u_i^\top$
- For the IFT mode response: the **ridged inverse** $(H + \delta I)^{-1}$ where $\delta$ is the PIRLS stabilization ridge

**The problem**: The cost function lives on the surface defined by $H_+^\dagger$, but the gradient is computed using $(H+\delta I)^{-1}$. These are different surfaces. The envelope theorem assumes a single consistent $H$.

**Questions**:
(a) Is there a single spectral regularization operator $\mathcal{R}(H)$ such that both $\log|\mathcal{R}(H)|$ and $\mathcal{R}(H)^{-1}$ are consistent (i.e., the gradient of $\log|\mathcal{R}(H)|$ w.r.t. $\rho$ uses the same operator as the IFT)?

(b) Options to consider:
  - **Soft clamping**: Replace $\sigma_i \to \max(\sigma_i, \epsilon)$ everywhere (both logdet and inverse). This makes $\mathcal{R}(H) = U\,\text{diag}(\max(\sigma_i, \epsilon))\,U^\top$. Is this smooth? Does it bias REML?
  - **Absolute value**: $\mathcal{R}(H) = U\,\text{diag}(|\sigma_i|)\,U^\top$. This is positive definite but discontinuous at $\sigma_i = 0$.
  - **Squared form**: Use $H^\top H$ instead of $H$ for the logdet. Then $\log|H^\top H|_+ = 2\log|H|_+$ when $H$ is PD, but is smooth through indefiniteness.

(c) In practice, how often does $H$ actually become indefinite? Is this only a problem for specific families (e.g., Gamma with log link near zero), or does it occur broadly?

### Q14: LAML under active inequality constraints

The LAML objective and its gradient assume **stationarity**: $\nabla_\beta F(\hat\beta, \theta) = 0$. This is the condition that makes the envelope theorem work and eliminates the $g_i^\top \beta_i$ cross-term from the profiled gradient.

When **inequality constraints** are active (e.g., monotonicity constraints forcing $\beta_j \geq 0$, or shape constraints), the KKT conditions replace stationarity:
$$\nabla_\beta F = A^\top\mu, \quad \mu \geq 0, \quad A\hat\beta \geq b, \quad \mu^\top(A\hat\beta - b) = 0$$

where $A$ is the constraint matrix, $b$ the bound vector, and $\mu$ the KKT multipliers.

**Consequences**:
- The mode $\hat\beta(\theta)$ is only **piecewise differentiable** in $\theta$ (it has kinks when constraints activate/deactivate)
- The envelope theorem no longer applies directly because $\nabla_\beta F \neq 0$
- The profiled gradient picks up an extra term: $\frac{\partial V}{\partial\theta_i}\bigg|_{\text{constrained}} = \frac{\partial V}{\partial\theta_i}\bigg|_{\text{unconstrained}} + \mu^\top A\beta_i$ (where $\beta_i$ is the mode response projected into the free subspace)

We currently handle this by **projecting** into the free subspace (the null space of active constraints), effectively ignoring the active directions. This gives a gradient that is correct within the active face but ignores the boundary effects.

**Questions**:
(a) Is the free-subspace projection a valid approximation for the LAML gradient? Under what conditions does it introduce bias? Specifically: if the true unconstrained optimum has $\hat\beta_j < 0$ but the constraint forces $\hat\beta_j = 0$, does the projected LAML gradient still point toward the correct smoothing parameters?

(b) A cleaner approach might be a **log-barrier LAML**: replace the hard constraint $\beta_j \geq 0$ with a barrier $-\mu\sum_j\log\beta_j$, profile out $\mu$ (or set it to a small constant), and compute LAML on the smooth barrier-augmented objective. Is there theory for the Laplace approximation under log-barrier penalization? Does the barrier term affect the $\log|H|$ correction?

(c) For **structural monotonicity** (I-spline constraints where coefficients must be non-negative to ensure monotonicity of the fitted function), the active set can change with $\theta$. The LAML surface has **ridges** at the $\theta$ values where a constraint activates. Does this cause convergence problems for Newton/BFGS, and is there a standard remedy?

### Q15: Exact Gaussian convolutions for non-logit link functions

When modeling binary outcomes with covariate measurement error, the inner loop must evaluate:
$$E_{\eta\sim N(\mu,\sigma^2)}\bigl[g^{-1}(\eta)\bigr] = \int_{-\infty}^{\infty} g^{-1}(\eta)\,\phi\bigl(\tfrac{\eta-\mu}{\sigma}\bigr)\,\frac{d\eta}{\sigma}$$

where $g^{-1}$ is the inverse link function and $\phi$ is the standard normal density.

For the **logit** link, $g^{-1}(\eta) = \text{sigmoid}(\eta)$, and the integral admits an exact, numerically stable representation via the Faddeeva function / erfcx series. We have implemented this and it gives smooth, kink-free derivatives to all orders.

For other link functions, we currently use **Gauss-Hermite quadrature** (7-21 points). This introduces:
- Numerical noise (piecewise-polynomial approximation of a smooth integral)
- Kinks in the outer LAML objective (because the GHQ nodes/weights don't move smoothly with $\mu$, $\sigma$)
- Difficulty computing higher-order derivatives (needed for $D_\beta H_L$ and $D^2_\beta H_L$)

**Questions**:
(a) For the **complementary log-log** (CLogLog) link, $g^{-1}(\eta) = 1 - \exp(-\exp(\eta))$, does the Gaussian convolution $\int (1 - e^{-e^\eta})\phi((\eta-\mu)/\sigma)\,d\eta/\sigma$ admit a closed-form series? The integrand involves a double exponential composed with a Gaussian — can this be expressed via Mellin-Barnes integrals, and if so, is the resulting series numerically stable and absolutely convergent?

(b) For the **probit** link, $g^{-1}(\eta) = \Phi(\eta)$, the Gaussian convolution is:
$$\int \Phi(\eta)\,\phi\bigl(\tfrac{\eta-\mu}{\sigma}\bigr)\,\frac{d\eta}{\sigma} = \Phi\Bigl(\frac{\mu}{\sqrt{1+\sigma^2}}\Bigr)$$
This is exact and trivial. But the **derivatives** with respect to $\mu$ and $\sigma$ — up to 4th order (needed for $D^2_\beta H_L$) — do they remain tractable? Write them out.

(c) For the **SAS (sinh-arcsinh)** link, $g^{-1}(\eta) = \Phi(\sinh(\epsilon^{-1}\sinh^{-1}(\eta) + \delta))$, the Gaussian convolution seems intractable analytically. Is there a practical middle ground between exact series and GHQ — e.g., adaptive quadrature with error bounds that guarantee smoothness of the outer objective?

(d) More generally: is there a **sufficient condition** on $g^{-1}$ that guarantees its Gaussian convolution admits a convergent series representation? (E.g., entire function of exponential type, or membership in a specific Hilbert space.)
