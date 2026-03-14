# Follow-Up: Gaps, Missing Derivations, and New Questions

Thank you for the comprehensive response. The unified $(a_i, g_i, B_i, \ell^S_i)$ abstraction and the master formulas (Sections 2–3) are exactly what we needed, and we are implementing them now.

However, several items were either **skipped entirely**, **answered incompletely** (missing the explicit formulas we need to implement), or **dismissed as too hard** when we actually need them. We also have **new questions** arising from the answers.

We emphasize again: **we need complete, explicit, implementable formulas**. "Use AD" or "use high-order quadrature" is not an acceptable final answer when a closed form or convergent series exists. We will implement whatever you derive, no matter how complex.

---

## Part A: Items That Were Skipped or Not Answered

### A1: D7 was not answered — Survival location-scale third-derivative correction

D7 asked for the explicit $D_\alpha H_L[u]$ for the **survival location-scale** model (Royston-Parmar with jointly estimated baseline hazard and scale). This was completely omitted from the response.

We need this for unification: survival location-scale currently has its own bespoke outer evaluation path. To route it through the unified evaluator, we need its `HessianDerivativeProvider` — which requires $D_\alpha H_L[u]$ and $D^2_\alpha H_L[u,v]$.

The model has joint coefficients $\alpha = (\beta_h, \beta_\sigma)$ with two linear predictors:
- $\eta_h = X_h \beta_h$ (log-cumulative-hazard basis coefficients)
- $\eta_\sigma = X_\sigma \beta_\sigma$ (log-scale)

The per-observation negative log-likelihood is:
$$-\ell_i = -\delta_i[\eta_{h,1i} + \log s_i - \eta_{\sigma,i}] + e^{\eta_{h,1i} - \eta_{\sigma,i}} - e^{\eta_{h,0i} - \eta_{\sigma,i}}$$

where $s_i = d_i^\top \beta_h$ is the log-hazard spline derivative, $\eta_{h,1i}$ and $\eta_{h,0i}$ are the exit/entry log-cumulative-hazards, and $\sigma_i = e^{\eta_{\sigma,i}}$ is the scale.

**Request**: Derive the full joint Hessian $H_L$ (all four blocks: $H_{hh}$, $H_{h\sigma}$, $H_{\sigma h}$, $H_{\sigma\sigma}$), then derive $D_\alpha H_L[u]$ showing all cross-block weight vectors $q_{ab}$ as functions of $(\eta_h, \eta_\sigma, y, \delta, s)$ and the perturbation $(u_h, u_\sigma)$. Then derive $D^2_\alpha H_L[u,v]$.

### A2: Q16c,d were not fully answered — MM fixed-point differentiation and which Hessian for $\log|H|$

**Q16c** asked: if $\hat\beta = T(\hat\beta; \theta)$ is the MM fixed-point, then by IFT on $\beta - T(\beta; \theta) = 0$:
$$\frac{d\hat\beta}{d\theta} = (I - \nabla_\beta T)^{-1} \nabla_\theta T$$

**We need**: the explicit formula for $\nabla_\beta T$ and $\nabla_\theta T$ when $T$ is the Charbonnier MM update. Specifically, the MM update solves:
$$[X^\top W_L X + A^\top W_{MM}(\beta^{(t)}) A]\,\beta^{(t+1)} = X^\top W_L z$$

where $W_L$ is the working likelihood weight and $z$ is the working response. At the fixed point $\beta^* = T(\beta^*; \theta)$, what is $(I - \nabla_\beta T)^{-1}$? Is it related to $H_{\text{true}}^{-1} H_{\text{MM}}$ where $H_{\text{true}}$ uses $D^2\psi$ and $H_{\text{MM}}$ uses $W_{MM}$?

**Q16d** asked: for the LAML $\frac{1}{2}\log|H|_+$ term, should $H$ use $D^2\psi$ (true penalty Hessian) or $W_{MM}$ (surrogate)? The response says the envelope theorem holds at convergence but doesn't answer this question. The issue is that $D^2\psi$ has the form $\text{diag}(\varepsilon^3/(t^2+\varepsilon^2)^{3/2})$ which can be very different from $W_{MM} = \text{diag}(\varepsilon/(2\sqrt{t^2+\varepsilon^2}))$ — especially near $t=0$ where $D^2\psi \to 1$ but $W_{MM} \to 1/(2\varepsilon)$.

**Request**: Which Hessian gives the correct Laplace approximation? If the true Hessian, derive the LAML gradient using the true Charbonnier Hessian (not the surrogate). If there's a correction factor between the two, derive it.

### A3: Q14b was not derived — Log-barrier LAML

The response mentions barrier regularization as "the cleanest way to restore differentiability" but doesn't derive the barrier-augmented Laplace correction.

**Request**: For
$$F_\tau(\beta, \theta) = F(\beta, \theta) - \tau \sum_{j \in \mathcal{C}} \log(\beta_j - b_j)$$

derive:
1. The barrier-augmented Hessian: $H_\tau = H + \tau \cdot \text{diag}(1/(\beta_j - b_j)^2)$ on the constrained coordinates.
2. The LAML objective: $V_\tau(\theta) = F_\tau(\hat\beta_\tau, \theta) + \frac{1}{2}\log|H_\tau| - \frac{1}{2}\log|S|_+ + c$
3. The profiled gradient $\partial V_\tau / \partial \rho_k$. Does the barrier term produce an extra contribution to the mode response $\beta_k = -H_\tau^{-1} g_k$, or does it only modify $H$ (and hence the traces)?
4. How should $\tau$ be chosen? Should it be profiled out (optimized jointly with $\theta$), held fixed, or annealed?

### A4: Profiled Gaussian Hessian under $\psi$ was not given

Section 9 gives the gradient for profiled Gaussian under $\psi$:
$$\partial_{\psi_j} V = \frac{a_{\psi_j}}{\hat\phi} + \frac{1}{2}\text{tr}(H^{-1}\dot{H}_{\psi_j}) - \frac{1}{2}\ell^S_{\psi_j}$$

But the **Hessian** $\partial^2 V / \partial\theta_i \partial\theta_j$ for profiled Gaussian was not derived. The chain rule through $\hat\phi(\theta) = D_p / \nu$ introduces additional terms beyond the fixed-dispersion LAML Hessian.

**Request**: Derive the complete profiled Gaussian Hessian $V_{ij}$ for general $\theta = (\rho, \psi)$. Specifically, the extra terms from:
$$\frac{\partial}{\partial\theta_j}\left[\frac{a_i}{\hat\phi}\right] = \frac{1}{\hat\phi}\left(\frac{\partial a_i}{\partial\theta_j} - \frac{a_i}{\hat\phi}\frac{\partial\hat\phi}{\partial\theta_j}\right)$$

Write out $\partial\hat\phi/\partial\theta_j$ in terms of the $(a_i, g_i, B_i)$ objects.

---

## Part B: Items Answered Incompletely

### B1: $M_i[u] = D_\beta B_i[u]$ for specific families — not derived for any family

The response correctly identifies $M_i[u]$ as the missing callback for the exact outer Hessian. Section 7.1 states it "can be written in terms of the same weight-derivative machinery" but then says "not from $W', W''$ alone; you also need the Jacobian-sensitivity pieces."

**We need the explicit formula for every family that will use $\psi$ coordinates.** Currently these are:

**(a) GLM with canonical link + moving design** (e.g., Binomial-logit where $X = X(\psi)$):

$B_{\psi_j} = \dot{X}_j^\top W X + X^\top W \dot{X}_j + X^\top \text{diag}(W' \odot \dot{X}_j \beta) X$

Then:
$$M_j[u] = D_\beta B_j[u] = \dot{X}_j^\top \text{diag}(c \odot Xu) X + X^\top \text{diag}(c \odot Xu) \dot{X}_j + X^\top \text{diag}(\ldots?) X$$

What is the "$\ldots$" in the third term? It involves $d_i \cdot (Xu)_i \cdot (\dot{X}_j\beta)_i + c_i \cdot (\dot{X}_j u)_i$. **Write this out completely** for Binomial-logit, Poisson-log, Gamma-log.

**(b) Gaussian location-scale with moving design:**

$B_{\psi_j}$ involves $\dot{X}_{\mu,j}$ and/or $\dot{X}_{\sigma,j}$. Since $W$ depends on $\sigma$ (and hence on $\beta_\sigma$), $B_j$ depends on $\beta$. Derive $M_j[u]$ for all four Hessian blocks.

**(c) General noncanonical exponential-dispersion family:**

Using the notation from Section 14 of your response ($h_1, h_2, h_3, V, V_1, V_2$), derive $M_j[u]$ in terms of these quantities plus $\dot{X}_j$.

### B2: $D^2_\beta H_L[u,v]$ for GAMLSS — weight vectors not given

Q9(c) says the second directional derivative needs $B^2 \times B(B+1)/2$ weight-like vectors, reduced by symmetry. But the actual formulas for these vectors were not given.

**Request**: For Gaussian location-scale (the simplest GAMLSS case), write out $D^2_\beta H_L[u,v]$ explicitly. The result should be a block matrix where each block is $X_a^\top \text{diag}(d_{ab}(u,v)) X_b$. Give each $d_{ab}$ as a function of $(\delta\mu^{(u)}, \delta\sigma^{(u)}, \delta\mu^{(v)}, \delta\sigma^{(v)}, r, \sigma)$.

Then generalize: for a $B$-block model with per-observation weight tensor $\partial^2 w_{ab}/\partial\eta_c \partial\eta_d$, write the general formula.

### B3: General noncanonical $d_\text{obs}$ (second weight derivative) — not given

Section 14 gives $c_\text{obs}$ for general noncanonical links via the auxiliary quantity $B_\eta$. But $d_\text{obs} = \partial c_\text{obs}/\partial\eta$ (needed for $D^2_\beta H_L[u,v]$) was not derived.

**Request**: Derive $d_\text{obs}$ for the general exponential-dispersion family with noncanonical link, in terms of $h_1, h_2, h_3, h_4$ (link derivatives up to 4th order) and $V, V_1, V_2, V_3$ (variance function derivatives up to 3rd order).

### B4: D2 missing families

The following families were not covered but are needed:

**(a) Binomial (not Bernoulli) with logit link**: $y_i \sim \text{Bin}(n_i, p_i)$, $p_i = \text{sigmoid}(\eta_i)$. Is it just $\omega \to \omega \cdot n_i$ in the Bernoulli formulas? If so, confirm. If not, derive.

**(b) Gaussian with log link**: $y_i \sim N(\mu_i, \phi)$, $\mu_i = e^{\eta_i}$. This is noncanonical. Derive $w, c, d$ (both observed and Fisher).

**(c) Gaussian with inverse link**: $\mu_i = 1/\eta_i$. Derive $w, c, d$.

**(d) Ordinal/multinomial**: For a proportional-odds model with $K$ categories, the per-observation Hessian is a matrix (not scalar). How does the correction formula $D_\beta H_L[u]$ generalize?

### B5: ALO computational details

The multi-predictor ALO formula (Section 16.3) gives:
$$\tilde\eta_i^{(-i)} \approx \hat\eta_i + (I_B - \mathcal{H}_{ii})^{-1} \mathcal{H}_{ii} W_i^{-1} s_i$$

But:
1. **When is $W_i$ singular?** For survival models with censored observations, certain Hessian blocks may be zero. What is the correct ALO formula when $W_i$ is singular?
2. **Efficient computation**: Computing $\mathcal{H}_{ii} = X_i H^{-1} X_i^\top W_i$ for each $i$ requires $B$ columns of $H^{-1}$. For $B=2$ and $n = 400,000$, this is $800,000$ linear solves. Is there a way to compute all leverages simultaneously via a single factorization, analogous to the diagonal of $X(X^\top W X + S)^{-1} X^\top W$ in the single-predictor case?
3. **ALO standard errors**: Can the ALO leverage be used to construct approximate leave-one-out standard errors for the linear predictor, analogous to Cook's distance? Derive the formula.

### B6: Stochastic trace estimation — missing practical details (N1)

The variance formula $\text{Var}(\hat{t}) = \frac{2}{M}\|\text{sym}(H^{-1}\dot{H}_k)\|_F^2$ is useful but:

1. **Rademacher vs Gaussian probes**: For structured $H$ (block-banded), Rademacher probes ($z \in \{-1, +1\}^p$) have $\text{Var} = \frac{2}{M}\sum_{i \neq j} a_{ij}^2$ which excludes diagonal terms. When $H^{-1}\dot{H}_k$ is diagonally dominant, Rademacher has lower variance. **Derive** the variance for Rademacher probes in our setting and give a concrete recommendation.

2. **Adaptive probe count**: We want to set $M$ adaptively based on a target relative error $\varepsilon$ in the gradient. Since $\|\text{sym}(H^{-1}\dot{H}_k)\|_F$ is unknown a priori, we need a running estimate. **Derive** an online estimator for the variance of $\hat{t}_k$ from the probe samples themselves (e.g., using the sample variance of $z_m^\top H^{-1}\dot{H}_k z_m$).

3. **Bias from truncated PCG**: When $H^{-1}z_m$ is computed via preconditioned conjugate gradients stopped at tolerance $\delta_\text{PCG}$, the solve error propagates into the trace estimate. **Bound** the bias $|\text{tr}(H^{-1}\dot{H}_k) - \text{tr}(\tilde{H}^{-1}\dot{H}_k)|$ in terms of $\delta_\text{PCG}$ and spectral properties of $H$.

---

## Part C: New Questions Arising from the Answers

### C1: The $r_\varepsilon$ spectral regularization — LAML gradient formula

Section 17.3 gives the smooth spectral regularization:
$$r_\varepsilon(\sigma) = \frac{1}{2}\left(\sigma + \sqrt{\sigma^2 + 4\varepsilon^2}\right)$$

This is excellent. But the LAML gradient uses $\text{tr}(H^{-1} \dot{H}_k)$, which assumes $H$ is used directly. When we replace $H$ with $\mathcal{R}_\varepsilon(H) = U \text{diag}(r_\varepsilon(\sigma_i)) U^\top$, the gradient becomes:

$$\frac{\partial}{\partial\rho_k}\log|\mathcal{R}_\varepsilon(H)| = \sum_i \frac{r'_\varepsilon(\sigma_i)}{r_\varepsilon(\sigma_i)} \cdot u_i^\top \dot{H}_k u_i$$

where $r'_\varepsilon(\sigma) = \frac{1}{2}(1 + \sigma/\sqrt{\sigma^2 + 4\varepsilon^2})$.

**Question**: Is this correct? If so, the trace formula becomes:
$$\text{tr}\left(\text{diag}\left(\frac{r'_\varepsilon(\sigma_i)}{r_\varepsilon(\sigma_i)}\right) U^\top \dot{H}_k U\right)$$

This is **not** the same as $\text{tr}(\mathcal{R}_\varepsilon(H)^{-1} \dot{H}_k)$ unless $r'_\varepsilon/r_\varepsilon = 1/r_\varepsilon$, which requires $r'_\varepsilon = 1$ (only true for large $\sigma$).

**Derive** the exact gradient and Hessian of $\log|\mathcal{R}_\varepsilon(H)|$ and $\mathcal{R}_\varepsilon(H)^{-1}$ in terms of the eigendecomposition and $\dot{H}_k$, and identify which standard formulas must be modified.

### C2: Second-order mode response — efficient computation

The Hessian algorithm (Section 10, Step 2) requires computing $\beta_{ij} = -H^{-1}(\ldots)$ for each pair $(i,j)$. With $q$ total hyperparameters, this is $O(q^2)$ linear solves (each $O(p^3)$ for dense, $O(p \cdot b^2)$ for sparse).

For large $q$ (e.g., 50 smoothing parameters + 5 $\psi$ coordinates = 55 total, giving 1540 pairs), this may dominate.

**Questions**:
(a) Can the solve for $\beta_{ij}$ be avoided by expressing the Hessian purely in terms of the first-order solves $\beta_i$? The term $g_i^\top H^{-1} g_j = g_i^\top \beta_j$ is free. The trace terms $\text{tr}(H^{-1}\ddot{H}_{ij})$ require $\beta_{ij}$ only through $C[\beta_{ij}]$ (which appears inside $\ddot{H}_{ij}$). Can this trace be rewritten to avoid materializing $\beta_{ij}$?

(b) Specifically: $\text{tr}(H^{-1} C[\beta_{ij}]) = \text{tr}(H^{-1} X^\top \text{diag}(c \odot X\beta_{ij}) X)$. If we define $T_m = (X H^{-1} X^\top)_{mm} \cdot c_m$ (a precomputable per-observation quantity), then:
$$\text{tr}(H^{-1} C[\beta_{ij}]) = \sum_m T_m \cdot (X\beta_{ij})_m$$

Can $X\beta_{ij}$ be computed without first computing $\beta_{ij}$? I.e., can we compute $X H^{-1} r_{ij}$ directly from the hat matrix?

### C3: Firth $D_\alpha I[u]$ for GAMLSS — explicit weight vectors

Section 16.1 gives the general structure $\dot{I}_i = I_i^{\text{fix}} + D_\alpha I[\alpha_i]$ but says for coupled blocks "you need the full joint $I^{-1}$ contraction" without deriving what $D_\alpha I[u]$ looks like.

For **Gaussian location-scale** with Fisher information:
$$I = J^\top W_F J$$
where $W_F$ is the Fisher weight (not observed). Then $D_\alpha I[u]$ has the same structure as $D_\alpha H_L[u]$ but with Fisher weights instead of observed weights.

**Request**: Confirm this or correct it. Then: for the Firth-corrected REML gradient, is the full formula:
$$V_k^{\text{Firth}} = V_k + \frac{1}{2}\text{tr}(I^{-1}\dot{I}_k)$$
where $\dot{I}_k$ uses the Fisher-information version of the correction? If so, the Firth gradient requires **both** the observed-information correction $C[u]$ (for the LAML terms) and the Fisher-information correction $C_F[u]$ (for the Firth term). Confirm.

### C4: $\psi$-dependent penalty logdet derivatives

The response (Section 4.2) defines $\ell^S_{\psi_j} = \partial_{\psi_j}\log|S|_+$ and $\ell^S_{\psi_i\psi_j} = \partial^2_{\psi_i\psi_j}\log|S|_+$.

For $S(\rho,\psi) = \sum_k \lambda_k S_k(\psi)$, these involve the derivative of a pseudo-logdeterminant of a sum of $\psi$-dependent matrices.

**Request**: Write out the explicit formula for $\ell^S_{\psi_j}$ in terms of $S$ and $\partial_{\psi_j} S$. Is it simply:
$$\ell^S_{\psi_j} = \text{tr}(S_+^{-1} \cdot \partial_{\psi_j} S)$$
where $S_+^{-1}$ is the pseudo-inverse restricted to the positive subspace? And the second derivative:
$$\ell^S_{\psi_i\psi_j} = \text{tr}(S_+^{-1} \partial^2_{\psi_i\psi_j} S) - \text{tr}(S_+^{-1} (\partial_{\psi_j} S) S_+^{-1} (\partial_{\psi_i} S))$$

Confirm or correct, and clarify what happens when $\partial_{\psi_j} S$ has components in the null space of $S$ (which would be the case if $\psi$ changes the null space structure).

### C5: Cross $(\rho_k, \psi_j)$ penalty logdet

Section 5 gives the cross-Hessian but uses $\ell^S_{kj}$ without deriving it.

For $S(\rho,\psi) = \sum_k \lambda_k S_k(\psi)$:
$$\partial_{\rho_k} \partial_{\psi_j} \log|S|_+ = \text{tr}\left(S_+^{-1} \lambda_k \partial_{\psi_j} S_k\right) - \text{tr}\left(S_+^{-1} A_k S_+^{-1} \partial_{\psi_j} S\right)$$

**Confirm or correct**. Also: in our implementation, penalty logdet derivatives for $\rho$ are precomputed from the penalty eigendecomposition. For $\psi$, they must be computed differently because $\partial_{\psi_j} S$ may not have the block-diagonal structure of $A_k$. What is the most efficient computation strategy?

### C6: When $S_k(\psi)$ shares null space but $\partial_{\psi_j} S_k$ does not

Section 17.1 says $\log|S|_+$ is smooth when all $S_k(\psi)$ share a fixed null space. But when $\psi$ moves the basis (e.g., anisotropy rotation), $\partial_{\psi_j} S_k$ may have components that lie partly in the null space of $S$.

**Question**: Does this cause the pseudo-logdet derivative $\ell^S_{\psi_j}$ to blow up or become ill-defined? If $S_+^{-1}$ is the pseudo-inverse and $\partial_{\psi_j} S$ has null-space components, the trace $\text{tr}(S_+^{-1} \partial_{\psi_j} S)$ drops those components (since $S_+^{-1}$ annihilates the null space from the left). Is this correct and well-defined, or does it miss a contribution?

### C7: Efficient Hessian for profiled Gaussian with many smoothing parameters

For profiled Gaussian REML, the standard EFS (extended Fellner-Schall) update avoids the full outer Hessian by using an approximate Newton step based on $\text{tr}(H^{-1} A_k H^{-1} A_l)$.

**Question**: With the unified $(\rho, \psi)$ framework, does EFS generalize? Specifically, can the EFS update formula:
$$\rho_k^{\text{new}} = \rho_k + \frac{\lambda_k \hat\beta^\top A_k \hat\beta - \text{tr}(H^{-1} A_k)}{\text{tr}(H^{-1} A_k H^{-1} A_k)}$$

be extended to $\psi_j$ coordinates by replacing $A_k$ with $B_{\psi_j}$ and $\lambda_k \hat\beta^\top A_k \hat\beta$ with $2a_{\psi_j}$? If not, what modifications are needed?

---

## Part D: Derivation Requests (New)

### D9: Survival location-scale full derivative provider

For the model in A1, derive the complete set:
1. $H_L$ (joint Hessian, all blocks)
2. $D_\alpha H_L[u]$ (first correction, all blocks and cross-terms)
3. $D^2_\alpha H_L[u,v]$ (second correction)
4. The per-observation weight vectors $w_{ab}$, $c_{abc}$ (first weight derivative), $d_{abcd}$ (second weight derivative) such that:
   $$[D_\alpha H_L[u]]_{ab} = X_a^\top \text{diag}\left(\sum_c c_{abc} \odot \delta\eta_c^{(u)}\right) X_b$$

### D10: Gaussian location-scale $D^2_\beta H_L[u,v]$ — all weight vectors

From B2: for Gaussian location-scale with $r = (y-\mu)/\sigma$, $\delta\mu^{(u)} = X_\mu u_\mu$, $\delta\sigma^{(u)} = X_\sigma u_\sigma$, derive all $d_{ab}(u,v)$ weight vectors for the 4 blocks of $D^2_\beta H_L[u,v]$:

$$[D^2_\beta H_L[u,v]]_{ab} = X_a^\top \text{diag}(d_{ab}(u,v)) X_b$$

There should be $2 \times 2 = 4$ blocks, each involving products of the two perturbations. Give explicit formulas.

### D11: $M_j[u]$ for Binomial-logit with moving design

For Binomial-logit with $W = n \cdot p(1-p)$, $c = n \cdot p(1-p)(1-2p)$, $d = n \cdot p(1-p)(1-6p+6p^2)$, and design $X(\psi)$ with $\dot{X}_j = \partial_{\psi_j} X$:

Derive $M_j[u] = D_\beta B_j[u]$ completely, where:
$$B_j = \dot{X}_j^\top \text{diag}(W) X + X^\top \text{diag}(W) \dot{X}_j + X^\top \text{diag}(c \odot \dot{X}_j\beta) X + \dot{X}_j^\top \text{diag}(c \odot \dot{X}_j\beta)\text{... ?}$$

Actually, first: write out $B_j$ completely for a non-canonical GLM with moving design. The third term above comes from $\partial_{\psi_j} W$, but $W$ depends on $\eta = X(\psi)\beta$, so $\partial_{\psi_j} W|_\beta = \text{diag}(c \odot \dot{X}_j\beta)$. But $\dot{X}_j$ also appears in the Hessian $X^\top W X$ through both $X$ factors. So $B_j$ should have **three** terms:
1. $\dot{X}_j^\top W X$ (left design moves)
2. $X^\top W \dot{X}_j$ (right design moves)
3. $X^\top \text{diag}(W' \odot \dot{X}_j\beta) X$ (weights change because $\eta$ changes)

Then $M_j[u] = D_\beta B_j[u]$ differentiates all three terms with respect to $\beta$ along direction $u$. The first two terms contribute via $D_\beta W$ (since $W = W(\eta)$ and $\eta = X\beta$). The third term contributes via $D_\beta(W' \odot \dot{X}_j\beta)$, which involves $W''$ and the product rule.

**Derive all terms explicitly.**

### D12: CLogLog Gaussian convolution — Mellin-Barnes expansion

The response says this is the Laplace transform of a lognormal and recommends high-order quadrature. We want more.

**Request**: Derive the Mellin-Barnes integral representation:
$$E[e^{-Y}] = \frac{1}{2\pi i}\int_{c-i\infty}^{c+i\infty} \Gamma(s) \cdot e^{-s\mu + s^2\sigma^2/2}\,ds$$

where $Y \sim \text{Lognormal}(\mu, \sigma^2)$. Then:

1. Give the saddle-point approximation of this integral (Laplace method on the Mellin-Barnes contour). What is the saddle point $s^*$ as a function of $(\mu, \sigma^2)$?
2. Give the asymptotic expansion to 3 terms beyond the saddle point. Is this sufficient for our purposes (relative error $< 10^{-8}$ when $\sigma < 2$)?
3. Derive the first 4 derivatives with respect to $\mu$ and $\sigma$ of the saddle-point approximation. These are what we need for $w, c, d$ under the CLogLog-Gaussian convolution.

If the saddle-point expansion is not accurate enough, give the **Gauss-Hermite quadrature with analytically differentiated weights** approach: i.e., use GHQ but differentiate the quadrature formula symbolically so that all derivatives are exact given the quadrature nodes.

---

## Part E: Clarifications

### E1: Symmetry of $C[\beta_i]\beta_j$

Section 3.1 states $C[\beta_i]\beta_j = C[\beta_j]\beta_i$ because "the third derivative tensor of a smooth scalar likelihood is symmetric." This is the statement that $\partial^3 \ell / \partial\beta_a \partial\beta_b \partial\beta_c$ is symmetric in all indices.

**But**: $C[u] = D_\beta H_L[u] = X^\top \text{diag}(c \odot Xu) X$. So $C[u]v = X^\top \text{diag}(c \odot Xu) Xv$. And $C[v]u = X^\top \text{diag}(c \odot Xv) Xu$. These are **not** the same vector in general ($X^\top(\text{diag}(a)b) \neq X^\top(\text{diag}(b)a)$ when $a \neq b$). What is true is that $u^\top C[v] w = v^\top C[w] u = w^\top C[u] v$ (the full contraction is symmetric), which implies $\beta_i^\top C[\beta_j] \beta_k$ is symmetric in all three indices.

But the claim is about the **matrix** $C[\beta_i]\beta_j$ equaling the **vector** $C[\beta_j]\beta_i$... these are different objects. $C[\beta_i]$ is a matrix, $C[\beta_i]\beta_j$ is a vector. $C[\beta_j]\beta_i$ is also a vector. Are these two vectors equal?

**Confirm**: Is the claim that $C[\beta_i]\beta_j = C[\beta_j]\beta_i$ (as vectors) true? If so, prove it. If not, how does this affect the second mode response formula?

### E2: The algorithm uses $C[\beta_i] \cdot \beta_j$ in Step 2 — clarify matrix vs vector

In the pseudocode (Section 10, Step 2):
```
rhs_ij = g_ij + B_i * beta_j + B_j * beta_i
       + 0.5 * (C(beta_i) * beta_j + C(beta_j) * beta_i)
```

$C(\beta_i)$ is a $p \times p$ matrix, $\beta_j$ is a $p$-vector. So $C(\beta_i) \cdot \beta_j$ is a $p$-vector (matrix-vector product). This is clear, but: is the 0.5 correct? The second mode response formula in Section 3.1 has:
$$\beta_{ij} = -H^{-1}(g_{ij} + B_i\beta_j + B_j\beta_i + C[\beta_i]\beta_j)$$

with **no factor of 1/2** and **no symmetrized** $C[\beta_j]\beta_i$ term. But the algorithm has both terms with 0.5. This is only consistent if $C[\beta_i]\beta_j = C[\beta_j]\beta_i$. **Resolve this discrepancy explicitly.**
