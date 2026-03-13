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
