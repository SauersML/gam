Below are the clean results. Two places where I think your current picture needs a real correction are:

1. for the survival outer Hessian, third-order chain terms in (q) are **not** enough once the (\log \sigma) block is present; you also need the 4th inner derivatives (q_{ssss}) and (q_{\vartheta sss});

2. the gauge zero-mode lives in the **joint** ((\rho_\kappa,\psi)) block, not in (H_{\psi\psi}) alone unless (\rho_\kappa) has already been profiled out.

I will use ( \Sigma = H^{-1}) below.

---

## 1. Binomial location-scale (m_1,m_2,m_3,m_4)

You want derivatives of
[
F(q)=-w{y\log G(q)+(1-y)\log(1-G(q))}.
]

### 1a. Logit link

Let
[
p = G(q)=\frac{1}{1+e^{-q}}, \qquad s=p(1-p).
]

Then everything collapses to the canonical Bernoulli forms:
[
m_1 = w(p-y),
]
[
m_2 = ws,
]
[
m_3 = ws(1-2p),
]
[
m_4 = ws(1-6p+6p^2)=ws(1-6s).
]

That is the best implementation form.

A numerically stable evaluation is:

* compute (p) with a branched expit,
  [
  p=
  \begin{cases}
  (1+e^{-q})^{-1}, & q\ge 0,[2mm]
  e^q(1+e^q)^{-1}, & q<0,
  \end{cases}
  ]
* then compute
  [
  s=p(1-p),
  \quad
  1-2p = -\tanh(q/2),
  ]
  so you can also write
  [
  m_3=-ws\tanh(q/2), \qquad m_4=ws(1-6s).
  ]

For tails, with (t=e^{-|q|}),
[
s=\frac{t}{(1+t)^2},
]
so (m_2,m_3,m_4) decay like (O(e^{-|q|})).

**Stability domain.** The naive one-sided formula (p=1/(1+e^{-q})) overflows only for (q\lesssim -709), but loss of significance in (1-p) or (p) by subtraction starts much earlier, around (|q|\gtrsim 36), where (e^{-|q|}) is already below machine epsilon. In that regime the derivatives are genuinely below machine precision anyway, so saturation to (0) is numerically acceptable.

---

### 1b. Complementary log-log / Gumbel link

Let
[
z=e^q,\qquad r=e^{-z}=1-G(q),\qquad p=G(q)=1-r= -\operatorname{expm1}(-z).
]

The clean derivative chain for the CDF itself is
[
G' = zr,
]
[
G'' = zr(1-z),
]
[
G''' = zr(z^2-3z+1),
]
[
G'''' = zr(-z^3+6z^2-7z+1).
]

So the exact “probit-analog” form is:

[
a=\frac{1-y}{r}-\frac{y}{p},
\qquad
a_1=\frac{1-y}{r^2}+\frac{y}{p^2},
]
[
a_2=\frac{2(1-y)}{r^3}-\frac{2y}{p^3},
\qquad
a_3=\frac{6(1-y)}{r^4}+\frac{6y}{p^4},
]
and with
[
g_1=zr,\quad
g_2=zr(1-z),\quad
g_3=zr(z^2-3z+1),\quad
g_4=zr(-z^3+6z^2-7z+1),
]
you have
[
m_1 = w,a,g_1,
]
[
m_2 = w,(a_1 g_1^2 + a g_2),
]
[
m_3 = w,(a_2 g_1^3 + 3a_1 g_1 g_2 + a g_3),
]
[
m_4 = w,(a_3 g_1^4 + 6a_2 g_1^2 g_2 + a_1(3g_2^2+4g_1g_3)+a g_4).
]

That is exact, but for code I would not use it directly.

#### Better implementation form

Define the stable ratio
[
h = \frac{z}{e^z-1}=\frac{z}{\operatorname{expm1}(z)}=\frac{zr}{p}.
]

Then the (y=1) derivatives are especially compact:
[
F_{y=1}' = -wh,
]
[
F_{y=1}'' = wh(h+z-1),
]
[
F_{y=1}''' = -wh\Big(2h^2+3(z-1)h+z^2-3z+1\Big),
]
[
F_{y=1}'''' = wh\Big(6h^3+12(z-1)h^2+(7z^2-18z+7)h+z^3-6z^2+7z-1\Big).
]

For (y=0), the cloglog loss is just (F_{y=0}=wz), so
[
F_{y=0}'=F_{y=0}''=F_{y=0}'''=F_{y=0}''''=wz.
]

Therefore for general (y\in[0,1]),
[
m_1 = w\big[(1-y)z-yh\big],
]
[
m_2 = w\big[(1-y)z+y,h(h+z-1)\big],
]
[
m_3 = w\Big[(1-y)z-y,h\big(2h^2+3(z-1)h+z^2-3z+1\big)\Big],
]
[
m_4 = w\Big[(1-y)z+y,h\big(6h^3+12(z-1)h^2+(7z^2-18z+7)h+z^3-6z^2+7z-1\big)\Big].
]

This is the form I would implement.

#### Stable domains and tail reformulations

For cloglog, the naive formulas break much earlier than logit.

* **Left tail** (q\ll 0): (z=e^q) is small and
  [
  p=1-e^{-z}
  ]
  suffers cancellation. Relative error is about (\varepsilon/z), so precision starts degrading noticeably once (z\lesssim 10^{-8}) ((q\lesssim -18)), and becomes poor by (q\lesssim -30). Use
  [
  p=-\operatorname{expm1}(-z),
  \qquad
  h=\frac{z}{\operatorname{expm1}(z)}.
  ]
  With `expm1`, no separate Taylor branch is strictly necessary. If you still want a series branch for tiny (z),
  [
  h = 1-\frac z2+\frac{z^2}{12}-\frac{z^4}{720}+O(z^6).
  ]

* **Right tail** (q\gg 0): (p) rounds to (1) once
  [
  r=e^{-z}<\varepsilon/2,
  ]
  i.e. (z\gtrsim 36.7), or (q\gtrsim 3.60). So computing (1-p) by subtraction is impossible there; compute (r=e^{-z}) directly.
  Also, formulas using (e^z) overflow once (z\gtrsim 709), i.e. (q\gtrsim \log 709 \approx 6.56). In that region use
  [
  h=\frac{zr}{1-r}\approx zr,
  ]
  which is safe because (r=e^{-z}) underflows gracefully to (0). For (y=1), all four derivatives are then effectively (0).

* **Loss itself** should be evaluated as
  [
  F(q)=w\Big[(1-y)e^q-y\log!\big(-\operatorname{expm1}(-e^q)\big)\Big],
  ]
  not via (\log p) and (\log(1-p)) from a naively computed (p).

---

## 2. No EFS-type fixed-point guarantee for indefinite (B_\psi)

Wood–Fasiolo’s ascent proof is built on a PSD direction with parameter-independent nullspace. In the non-Gaussian extension they explicitly note that once the Hessian depends on smoothing parameters, the exact maximization guarantee is lost; their theorem uses the expected Hessian (or an observed Hessian that is itself positive definite) precisely to recover the needed cone structure. ([ar5iv][1])

So for your design-moving coordinates ( \psi_d ), the answer is **(b)**:

### There is no universally convergent EFS-type fixed-point map for indefinite (B_d) based only on

[
a_d,\qquad \operatorname{tr}(H^{-1}B_d),\qquad \operatorname{tr}(H^{-1}B_dH^{-1}B_e),\qquad v_d.
]

A concrete counterexample is enough.

Take a scalar coordinate (\psi), set (H(0)=I_2), and let
[
B=\begin{pmatrix}1&0\0&-1\end{pmatrix}
]
(indefinite). Consider the local family
[
V_c(\psi)=a,\psi+\frac12\log\det!\Big(I_2+\psi B+\frac12 c\psi^2 I_2\Big).
]

At (\psi=0),
[
\partial_\psi V_c(0)=a+\frac12\operatorname{tr}(B)=a,
]
[
\operatorname{tr}(H^{-1}BH^{-1}B)=\operatorname{tr}(B^2)=2,
]
and these are **independent of (c)**. So any fixed-point map that only uses the allowed quantities produces the same nonzero step (\Delta) for every (c).

But
[
V_c''(0)=c-1.
]
Hence
[
V_c(\Delta)-V_c(0)=a\Delta+\frac12(c-1)\Delta^2+O(\Delta^3).
]
By choosing (c) large positive or large negative, the same step can be made ascent or descent. Therefore no update rule based only on those local invariants can have a uniform convergence or ascent guarantee.

### What structural property is necessary and sufficient?

For an EFS/Fellner–Schall style proof, the needed property is:

* after a possible sign flip, the coordinate must move the criterion along a **cone-preserving PSD direction**,
* and the associated nullspace must stay fixed.

Operationally, this is the same condition as
[
H^{-1/2}B_dH^{-1/2}\succeq 0
\quad\text{(or }\preceq 0\text{)}
]
together with a parameter-independent nullspace on the penalized subspace.

That condition is sufficient because it restores the Loewner-order inequality used in the Wood–Fasiolo theorem; it is also essentially necessary for any proof of the same type, because mixed inertia destroys the ordering that makes the scalar update monotone. The positivity of
[
\operatorname{tr}(H^{-1}BH^{-1}B)=|H^{-1/2}BH^{-1/2}|_F^2
]
is not enough: it is only a norm, not a curvature bound.

### Practical replacement

A cheap safe replacement is not a fixed-point method but a **safeguarded preconditioned gradient/Newton step** on the constrained (\psi)-subspace:
[
\Delta\psi = -\alpha, G^+ g_\psi,
\qquad
G_{de}=\operatorname{tr}(H^{-1}B_dH^{-1}B_e),
]
with backtracking or trust region. This uses the same trace Gram matrix, stays cheap, and avoids pretending that the Gram denominator is the true scalar curvature.

---

## 3. Observed vs expected information in the outer REML/LAML

Wood et al. write the exact LAML in terms of the **negative Hessian of the penalized log-likelihood at the mode**, while Tierney–Kadane formulate Laplace/TK using the **observed information at the relevant maximizer**. Wood–Fasiolo then treat the expected Hessian as a PQL-type simplification in settings where exact Hessian dependence is inconvenient. ([School of Mathematics][2])

### (a) Consistency of (\hat\theta)

For a regular model with fixed outer dimension, both (H_{\text{obs}}) and (H_{\text{Fisher}}) target the same pseudo-true smoothing/link/anisotropy parameter to first order.

Reason: at the mode,
[
H_{\text{obs}} = H_{\text{Fisher}} + \Delta_n,
\qquad
\Delta_n = O_p(\sqrt n),
]
while (H=O(n)). Therefore
[
\log|H_{\text{obs}}|-\log|H_{\text{Fisher}}|
= \operatorname{tr}(H^{-1}\Delta_n)+O_p(|H^{-1}\Delta_n|^2)
= O_p(n^{-1/2})
]
for fixed (p), and similarly for the outer gradient. So the two outer criteria differ by a lower-order perturbation and have the same first-order minimizer and rate, provided the usual Laplace regularity holds. With growing coefficient dimension you need the same kind of high-dimensional control used for LAML itself; Wood’s consistency argument assumes (\dim(\beta)=O(n^\alpha)) with (\alpha<1/3), and the high-dimensional Laplace literature shows the error can scale like (O(p^3/n)) in linear exponential-family structure and worse in general models. ([School of Mathematics][2])

So: **same asymptotic target to first order; same rate under the same growth conditions that already justify Laplace**.

### (b) Which one gives a valid Laplace approximation?

**Observed information.**

The Laplace approximation to
[
\int \exp{-F(\beta)},d\beta
]
comes from Taylor expansion of the actual integrand around the actual mode, so the quadratic term is the actual Hessian
[
H_{\text{obs}}=\nabla^2_{\beta\beta}F(\hat\beta,\theta).
]
Replacing it by expected Fisher information changes the quadratic approximation and therefore changes the approximation itself. That replacement can be asymptotically equivalent, but it is not the exact Laplace approximation of the stated integral. Tierney–Kadane’s formulation is explicit on this point: the required quantity is the observed information at the maxima. ([McGill Math][3])

### (c) Mixed strategy

Yes, there is a mixed strategy that is both stable and Laplace-valid:

* use **Fisher scoring** or any other stable inner solver to find (\hat\beta(\theta));
* once converged, recompute **(H_{\text{obs}})** and use that in
  [
  \frac12\log|H|,
  \quad
  \frac12\operatorname{tr}(H^{-1}B_k),
  \quad
  \text{and TK}.
  ]

That is fine because the inner algorithm is only a way to get the mode. The outer criterion should still use the observed curvature at that mode. The opposite hybrid — observed Newton internally, but expected Fisher in the outer determinant/trace — is not the exact Laplace criterion; it is a quasi-Laplace/PQL surrogate. ([School of Mathematics][2])

### (d) Learnable links

The conclusion does **not** change qualitatively.

When the link itself is estimated, the difference between observed and expected curvature can be larger in finite samples because the link-shape coordinates affect both the mean map and the Jacobian/design terms, so the observed Hessian carries extra residual-dependent pieces. But the exact Laplace approximation is still built from the observed penalized Hessian at the mode; Fisher remains only an asymptotically equivalent surrogate under regularity. So for learned links the recommendation is the same:

* Fisher/expected is acceptable as an inner scoring metric or preconditioner;
* observed is the correct outer (H) for LAML, its trace derivatives, and TK.

---

## 4. Sum-to-zero anisotropy: projection vs reparameterization

Let
[
\mathcal M = {\psi\in\mathbb R^D: 1^\top\psi=0},
\qquad
P = I-\frac1D 11^\top.
]

### (a) Unconstrained optimize + post-hoc projection?

Not in general.

If gauge invariance is exact, the invariant direction is the **joint** direction
[
(\delta\rho_\kappa,\delta\psi)=(-c,; c1).
]
So the unconstrained problem has a ridge in the coupled ((\rho_\kappa,\psi)) block. An unconstrained optimizer can drift along that ridge or become ill-conditioned there. Post-hoc replacing
[
\psi \leftarrow \psi-\bar\psi,1
]
is only model-equivalent if you also shift the isotropic scale coordinate:
[
\rho_\kappa \leftarrow \rho_\kappa+\bar\psi,
\qquad
\bar\psi=\frac1D 1^\top\psi.
]

Projecting (\psi) alone changes the model.

So:

* in exact arithmetic, unconstrained optimization may land anywhere on the ridge;
* post-hoc projection recovers the constrained representative **only if** (\rho_\kappa) is adjusted simultaneously;
* numerically, direct constrained optimization is better because it removes the flat direction during optimization rather than after it.

### (b) Are full-space derivatives correct on the manifold?

Yes, after tangent projection.

For any ambient gradient (g_\psi=\nabla_\psi V), the intrinsic gradient on (\mathcal M) under the standard Euclidean metric is
[
\nabla_{\mathcal M} V = P g_\psi.
]
Reason: every tangent vector (\delta\psi) satisfies (1^\top\delta\psi=0), hence
[
dV[\delta\psi]=g_\psi^\top\delta\psi=(Pg_\psi)^\top\delta\psi.
]

Likewise, because (\mathcal M) is an affine subspace, the intrinsic Hessian is just the restriction of the ambient Hessian:
[
H_{\mathcal M} = P H_{\psi\psi} P
\quad\text{on }1^\perp.
]

So your full-space derivatives are fine; the manifold versions are obtained by projection.

### (c) Which pseudoinverse?

For the constrained (\psi)-problem, the right second-order step is
[
\delta\psi
==========

-,C(C^\top H_{\psi\psi} C)^{-1} C^\top g_\psi,
]
or equivalently
[
\delta\psi = -(P H_{\psi\psi} P)^+ P g_\psi
]
if you insist on staying in ambient coordinates.

That is the ordinary Moore–Penrose pseudoinverse on the tangent space. There is no extra “REML pseudoinverse” unless you deliberately choose a non-Euclidean metric.

The important correction is this: the genuine gauge zero-mode belongs to the **coupled** block ((\rho_\kappa,\psi)), not to (H_{\psi\psi}) alone unless (\rho_\kappa) has already been eliminated. So the cleanest Newton system is either:

* work in reduced coordinates ((\rho_\kappa,\tilde\psi)), or
* solve the KKT system
  [
  \begin{bmatrix}
  H_{\psi\psi} & 1\
  1^\top & 0
  \end{bmatrix}
  \binom{\delta\psi}{\lambda}
  = -
  \binom{g_\psi}{0}.
  ]

If you optimize jointly in ((\rho_\kappa,\psi)), then the null vector is proportional to
[
u = (-1,1,\dots,1),
]
and the pseudoinverse should be applied to the full coupled block after gauge fixing.

### (d) Explicit reparameterization

Let (C\in\mathbb R^{D\times(D-1)}) satisfy
[
C^\top 1=0.
]

The simplest correct choice is any **Euclidean-orthonormal** basis of (1^\perp):
[
C^\top C = I_{D-1},\qquad C^\top 1=0.
]
A Helmert contrast matrix is ideal.

Then write
[
\psi = C\tilde\psi.
]
The reduced derivatives are
[
g_{\tilde\psi} = C^\top g_\psi,
\qquad
H_{\tilde\psi\tilde\psi}=C^\top H_{\psi\psi}C.
]

If you want a metric-aware basis for some SPD metric (M) on (\psi)-space, start from any orthonormal (Q) with (Q^\top1=0), then set
[
R^\top R = Q^\top M Q,\qquad C = Q R^{-1}.
]
Then
[
C^\top M C = I,
\qquad
C^\top 1 = 0.
]

That is the right way to “respect the metric”. But for ordinary Newton on the constrained parameter space, plain orthonormal Helmert contrasts are enough.

---

## 5. Tierney–Kadane / fully exponential correction in multiblock form

Tierney–Kadane give the fixed-dimension second-order correction viewpoint, and the high-dimensional literature shows the leading correction stays (O(n^{-1})) only when dimension growth is controlled; otherwise the constants depend on (p). ([McGill Math][4])

Let (F(\beta)) be the penalized objective, (\hat\beta) its mode, and (\Sigma=H^{-1}). Write
[
T_{\alpha\beta\gamma} = \partial_{\alpha\beta\gamma}^3 F(\hat\beta),
\qquad
Q_{\alpha\beta\gamma\delta} = \partial_{\alpha\beta\gamma\delta}^4 F(\hat\beta),
]
with compound indices (\alpha,\beta,\dots) running over **all coefficients in all blocks**.

Then the first correction to the **log-integral** is
[
TK
==

-\frac18 Q_{\alpha\beta\gamma\delta}\Sigma_{\alpha\beta}\Sigma_{\gamma\delta}
+\frac1{12}T_{\alpha\beta\gamma}T_{\mu\nu\lambda}
\Sigma_{\alpha\mu}\Sigma_{\beta\nu}\Sigma_{\gamma\lambda}
+\frac18 T_{\alpha\beta\gamma}T_{\mu\nu\lambda}
\Sigma_{\alpha\beta}\Sigma_{\gamma\mu}\Sigma_{\nu\lambda}.
]

This is just the Gaussian-moment simplification of
[
-\frac1{24},E_\Sigma[Q(Z,Z,Z,Z)]
+\frac1{72},E_\Sigma[T(Z,Z,Z)^2].
]

### (a) Block structure

Nothing changes in the multiblock case except the indexing. Treat the block label as part of the index.

So for (\beta=(\beta_1,\dots,\beta_B)), with (B=2,3,4), the same formula holds with compound indices
[
\alpha=(r,a),\quad r\in{1,\dots,B}.
]

There is **no blockwise factorization** unless both:

* the Hessian inverse (\Sigma) is block diagonal, and
* the derivative tensors (T,Q) vanish whenever block labels mix.

That is not true in coupled GAMLSS or survival models, so cross-block contractions are part of TK, not an extra afterthought.

### (b) What cross-block terms appear?

All block combinations already present in (T) and (Q).

For GAMLSS with blocks ((\mu,\sigma)), third-order tensors such as
[
T^{\mu\mu\sigma},\quad T^{\mu\sigma\sigma}
]
and fourth-order tensors such as
[
Q^{\mu\mu\mu\sigma},\quad Q^{\mu\mu\sigma\sigma},\quad Q^{\mu\sigma\sigma\sigma}
]
feed directly into the contractions above.

For the 4-block survival model, exactly the same rule applies: if the observationwise negative log-likelihood has nonzero mixed predictor derivatives, then the corresponding block-mixed (T)- and (Q)-blocks appear in TK automatically. There are no extra multiblock TK terms beyond these tensor contractions.

### (c) Derivative (\partial TK/\partial\theta_k)

Let
[
\dot\Sigma_k = \partial_k \Sigma = -\Sigma B_k \Sigma,
]
where (B_k) is the **total** Hessian drift for (\theta_k), including fixed-(\beta) and IFT pieces.

Let
[
\dot T_k = \partial_k T,\qquad \dot Q_k=\partial_k Q
]
also be total derivatives evaluated at the mode.

Then differentiating the contraction formula gives
[
\partial_k TK
=============

-\frac18 \dot Q_{k,\alpha\beta\gamma\delta}\Sigma_{\alpha\beta}\Sigma_{\gamma\delta}
+\frac14 Q_{\alpha\beta\gamma\delta}(\Sigma B_k\Sigma)*{\alpha\beta}\Sigma*{\gamma\delta}
]
[
\quad
+\frac16 \dot T_{k,\alpha\beta\gamma}T_{\mu\nu\lambda}
\Sigma_{\alpha\mu}\Sigma_{\beta\nu}\Sigma_{\gamma\lambda}
-\frac14 T_{\alpha\beta\gamma}T_{\mu\nu\lambda}
(\Sigma B_k\Sigma)*{\alpha\mu}\Sigma*{\beta\nu}\Sigma_{\gamma\lambda}
]
[
\quad
+\frac14 \dot T_{k,\alpha\beta\gamma}T_{\mu\nu\lambda}
\Sigma_{\alpha\beta}\Sigma_{\gamma\mu}\Sigma_{\nu\lambda}
-\frac38 T_{\alpha\beta\gamma}T_{\mu\nu\lambda}
(\Sigma B_k\Sigma)*{\alpha\beta}\Sigma*{\gamma\mu}\Sigma_{\nu\lambda}.
]

Because of symmetry, in each line I only wrote one representative placement of ((\Sigma B_k\Sigma)); the equal placements have already been counted in the coefficients (1/4) and (3/8).

That is the correct multiblock formula. The only model-specific work is inside (\dot T_k) and (\dot Q_k).

### (d) Is TK worth computing at biobank scale?

With fixed (p), yes TK is the standard (O(n^{-1})) correction to the **log integral**. But once (p) grows, the correction is no longer usefully described as “(O(1/n))” unless the dimension dependence is tiny. High-dimensional Laplace results suggest relative errors like (O(p^3/n)) in linear exponential-family structure and worse in more general settings. ([McGill Math][4])

So the right answer is:

* if your effective latent dimension stays modest, or the penalization makes the relevant contractions very small, TK is often negligible;
* with raw block sizes around (10^3) per block, it is **not** safe to assume (|TK|) is automatically negligible just because (n) is (4\times 10^5).

I would drop TK only after checking the size of the three explicit contractions above on a few real fits. The criterion is not (n) alone; it is the size of those contractions.

---

## 6. Survival chain-rule completeness for the outer REML Hessian

To avoid your notation clash, let me rename the predictor blocks:

* ( \tau ): time / baseline-transform block (enters (h) and (g=d\eta_\tau/dt)),
* ( \vartheta ): threshold/location block,
* ( s ): log-scale block,
* ( w ): wiggle block.

Write
[
u = q(\vartheta,s,w)-h(\tau),
\qquad
q(\vartheta,s,w) = -\vartheta e^{-s}+w.
]

For each entry/exit contribution, let (m_r = d^rF/du^r) for that scalar contribution.

### (a) Do 3rd-order chain terms in (q) suffice?

No. For the **outer Hessian**, you need the 4th derivative of the composed likelihood with respect to predictor blocks, and that requires the 4th inner derivatives of (u) wherever they are nonzero.

The generic scalar-inner composition formulas are:

[
F_{\alpha\beta}
===============

m_2 u_\alpha u_\beta + m_1 u_{\alpha\beta},
]

[
F_{\alpha\beta\gamma}
=====================

m_3 u_\alpha u_\beta u_\gamma
+
m_2\big(
u_{\alpha\beta}u_\gamma
+
u_{\alpha\gamma}u_\beta
+
u_{\beta\gamma}u_\alpha
\big)
+
m_1 u_{\alpha\beta\gamma},
]

[
F_{\alpha\beta\gamma\delta}
===========================

m_4 u_\alpha u_\beta u_\gamma u_\delta
+
m_3!!\sum_{\text{6 perms}}!u_{\alpha\beta}u_\gamma u_\delta
+
m_2!!\sum_{\text{3 perms}}!u_{\alpha\beta}u_{\gamma\delta}
+
m_2!!\sum_{\text{4 perms}}!u_{\alpha\beta\gamma}u_\delta
+
m_1 u_{\alpha\beta\gamma\delta}.
]

So the distinct chain-rule product types are exactly:

1. (m_4,u_\alpha u_\beta u_\gamma u_\delta)
2. (m_3,u_{\alpha\beta}u_\gamma u_\delta)
3. (m_2,u_{\alpha\beta}u_{\gamma\delta})
4. (m_2,u_{\alpha\beta\gamma}u_\delta)
5. (m_1,u_{\alpha\beta\gamma\delta})

Now list the nonzero inner derivatives of (u).

First derivatives:
[
u_\tau = -h_\tau,
\qquad
u_\vartheta = -e^{-s}=-\sigma^{-1},
\qquad
u_s = \vartheta e^{-s}=\vartheta/\sigma,
\qquad
u_w = 1.
]

Second derivatives:
[
u_{\vartheta s}=u_{s\vartheta}=\sigma^{-1},
\qquad
u_{ss}=-\vartheta/\sigma.
]

Third derivatives:
[
u_{\vartheta ss}=u_{s\vartheta s}=u_{ss\vartheta}=-\sigma^{-1},
\qquad
u_{sss}=\vartheta/\sigma.
]

Fourth derivatives:
[
u_{\vartheta sss}=u_{s\vartheta ss}=u_{ss\vartheta s}=u_{sss\vartheta}=\sigma^{-1},
\qquad
u_{ssss}=-\vartheta/\sigma.
]

Everything else vanishes if (h(\tau)) is linear in the time predictor.

So the correct conclusion is:

* (m_4) is enough on the likelihood side;
* but on the **inner chain-rule side** you also need
  [
  u_{\vartheta sss},\qquad u_{ssss}.
  ]

So the statement “3rd-order chain derivatives through (s^3) suffice” is false whenever a 4th predictor derivative with four (s)-slots, or one (\vartheta)-slot plus three (s)-slots, can appear. In particular, a pure ((s,s,s,s)) or mixed ((\vartheta,s,s,s)) block contribution to (Q[v_k,v_l]) needs those 4th inner derivatives.

### (b) Time-block entry/exit decomposition

Write the negative log-likelihood per observation as
[
F_i
===

F_i^{(1)}(u_{1,i})
+
F_i^{(0)}(u_{0,i})
------------------

w_i d_i \log g_i,
]
where:

* (F_i^{(1)}) is the exit contribution from (d_i\log f(u_1)+(1-d_i)\log S(u_1)),
* (F_i^{(0)}) is the entry contribution from (-\log S(u_0)),
* (\log g_i) is time-only.

Let (m^{(1)}_r) and (m^{(0)}_r) be the (r)-th (u)-derivatives of those two scalar pieces.

Then the cross-block Hessian blocks are additive over entry and exit:
[
H_{\tau\vartheta}
=================

X_\tau^\top
\operatorname{diag}!\big(
m^{(1)}*2 u*{1,\tau}u_{1,\vartheta}
+
m^{(0)}*2 u*{0,\tau}u_{0,\vartheta}
\big)
X_\vartheta,
]
[
H_{\tau s}
==========

X_\tau^\top
\operatorname{diag}!\big(
m^{(1)}*2 u*{1,\tau}u_{1,s}
+
m^{(0)}*2 u*{0,\tau}u_{0,s}
\big)
X_s,
]
because (u_{\tau\vartheta}=u_{\tau s}=0).

Their first and second outer drifts come from the corresponding 3rd- and 4th-order predictor derivatives:
[
F^{(j)}*{\tau\vartheta r},
\qquad
F^{(j)}*{\tau\vartheta rs},
\qquad
F^{(j)}*{\tau s r},
\qquad
F^{(j)}*{\tau s rs},
\qquad j\in{0,1},
]
using the generic formulas above.

Since higher time-predictor derivatives vanish, the only nontrivial lower-order inner terms in those expressions come from
[
u_{\vartheta s},\ u_{ss},\ u_{\vartheta ss},\ u_{sss},\ u_{\vartheta sss},\ u_{ssss}.
]
So:

* **entry and exit contributions are separate and additive**;
* **all time–threshold and time–logscale cross terms come only from the (u_0,u_1) pieces**;
* the ( \log g ) term contributes only to time–time blocks.

That is the correct decomposition.

### (c) Monotonicity barrier

For the barrier
[
B_i(g_i) = -\tau \log g_i,
\qquad
g_i = X_{\tau,\mathrm{deriv},i}\beta_\tau,
]
the predictor (g_i) is linear in (\beta_\tau). Therefore
[
B_i''(g)=\tau g^{-2},
\qquad
B_i'''(g)=-2\tau g^{-3},
\qquad
B_i''''(g)=6\tau g^{-4}.
]

So the barrier behaves exactly like an ordinary 1-predictor likelihood term on the time block.

Hence:

* if (X_{\tau,\mathrm{deriv}}) does **not** move with outer (\theta), then the barrier has no fixed-(\beta) outer drift at all, and its contribution enters the outer gradient/Hessian only through the standard IFT pieces (C[v_k]) and (Q[v_k,v_l]);
* if (X_{\tau,\mathrm{deriv}}) does move with (\theta), then you just add the ordinary design-motion terms. There is still no separate barrier-specific outer calculus.

So your statement is correct, with that design-motion caveat.

---

## 7. Moving-nullspace correction for (\log|S|_+)

Holbrook’s paper is useful here for the basic geometry: the pseudodeterminant is discontinuous across rank changes, and its differential is clean only along directions that preserve the kernel. ([Andrew Holbrook][5])

Assume from here on:

* (S(\theta)) is symmetric PSD,
* the nullity is constant in the neighborhood,
* there is a spectral gap between the positive eigenvalues and (0).

Write
[
S = U_+\Sigma_+ U_+^\top,
\qquad
L_k = U_+^\top S_k U_0,
\qquad
S_k=\partial_k S,
\qquad
S_{kl}=\partial_{kl}^2 S.
]

### (a) Is the leakage term complete?

Yes — under the constant-rank/gap assumptions, your formula is the complete exact bilinear second derivative:
[
\partial_{kl}^2 \log|S|_+
=========================

## \operatorname{tr}(S_+^{-1}S_{kl})

\operatorname{tr}(S_+^{-1}S_l S_+^{-1}S_k)
+
2,\operatorname{tr}(\Sigma_+^{-2}L_kL_l^\top).
]

No further nullspace-rotation terms appear at second order.

Why: the only way nullspace motion affects the positive eigenvalues at second order is through first-order mixing of (U_+) into (U_0). Rayleigh–Schrödinger perturbation theory gives exactly one such contribution, quadratic in the leakage matrix, and that is precisely the (2,\operatorname{tr}(\Sigma_+^{-2}L_kL_l^\top)) term.

So your leakage correction is complete.

### (b) Does the gradient need a nullspace correction?

On a constant-rank manifold: **no**.

The exact first derivative is
[
\partial_k \log|S|*+ = \operatorname{tr}(S*+^{-1}S_k).
]

Reason: first-order changes in the positive eigenvalues depend only on the Rayleigh quotients inside the positive eigenspace. Positive–null leakage changes eigenvectors at first order, but affects positive eigenvalues only at second order.

So there is no first-order leakage correction.

If rank is not constant, then (\log|S|_+) is not differentiable there anyway. In that case there is no finite correction term to add; the problem is genuine nonsmoothness, not a missing gradient term. Holbrook’s discontinuity example makes that point sharply. ([Andrew Holbrook][5])

### (c) What about the (\varepsilon)-threshold implementation?

Yes, the hard threshold introduces bias and artificial kinks.

If you define “null” by (\sigma_i\le\varepsilon), then as (\theta) varies an eigenvalue crossing (\varepsilon) makes the objective, gradient, and Hessian jump discretely. That is not an intrinsic feature of the pseudologdet; it is an artifact of the truncation rule.

The clean smooth relaxation is
[
L_\delta(S)=\log\det(S+\delta I)-m_0\log\delta,
]
where (m_0) is the known nullity of the penalty family.

Then
[
\partial_k L_\delta
===================

\operatorname{tr}\big((S+\delta I)^{-1}S_k\big),
]
[
\partial_{kl}^2 L_\delta
========================

## \operatorname{tr}\big((S+\delta I)^{-1}S_{kl}\big)

\operatorname{tr}\big((S+\delta I)^{-1}S_l(S+\delta I)^{-1}S_k\big).
]

This is (C^\infty) in (\theta). As (\delta\downarrow0), it converges to the true pseudo-logdet and its derivatives on any constant-rank neighborhood, and the leakage term is recovered automatically in that limit.

For Duchon/Matérn penalties, the nullity is usually known from the polynomial nullspace and should remain fixed under anisotropic rescaling. In that setting (L_\delta) is the right regularization. If the nullity can genuinely change, then the exact pseudo-logdet is inherently nonsmooth and no unbiased smooth exact replacement exists.

---

That gives the mathematical pieces I would treat as settled:

* **Deliverable 1:** logit closed forms are canonical; cloglog is best coded through (z=e^q) and (h=z/\expm1(z)).
* **Deliverable 2:** no general EFS-type fixed-point guarantee for indefinite (B_\psi); PSD + fixed nullspace is the crucial structural property.
* **Deliverable 3:** observed (H) is the exact Laplace choice; Fisher is asymptotically equivalent but only a surrogate.
* **Deliverable 4:** constrain during optimization; if you project afterward, also shift (\rho_\kappa).
* **Deliverable 5:** multiblock TK is just the full-tensor contraction on compound indices; cross-block terms are already inside (T,Q,\Sigma).
* **Deliverable 6:** for the outer Hessian you do need (u_{\vartheta sss}) and (u_{ssss}).
* **Deliverable 7:** your leakage Hessian term is complete; the gradient needs no leakage term on constant-rank neighborhoods; hard (\varepsilon)-thresholding should be replaced by a smooth (\delta)-regularization.

[1]: https://ar5iv.org/pdf/1606.04802 "https://ar5iv.org/pdf/1606.04802"
[2]: https://webhomes.maths.ed.ac.uk/~swood34/gsm.pdf "https://webhomes.maths.ed.ac.uk/~swood34/gsm.pdf"
[3]: https://www.math.mcgill.ca/dstephens/680/Handouts/OldPDFs/TierneyKadane-1986-JASA.pdf?utm_source=chatgpt.com "Accurate Approximations for Posterior Moments and Marginal Densities"
[4]: https://www.math.mcgill.ca/dstephens/680/Handouts/OldPDFs/TierneyKadane-1986-JASA.pdf "https://www.math.mcgill.ca/dstephens/680/Handouts/OldPDFs/TierneyKadane-1986-JASA.pdf"
[5]: https://andrewjholbrook.github.io/papers/pseudo.pdf "Differentiating the pseudo determinant"

