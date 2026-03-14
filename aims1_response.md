Below is the cleanest way to unify the outer optimizer.

The short mathematical answer is:

1. **Yes** — you can fold (\psi) into the same evaluator as (\rho) by replacing “penalty-root derivative (A_k)” with a generic per-coordinate package of fixed-(\beta) derivatives. That is exactly the right abstraction boundary.
2. **But** the current 2-method provider is only sufficient for the exact Hessian when the fixed-(\beta) Hessian drift is itself (\beta)-independent. That is true for pure (\rho), but not generally for (\psi). So the unified evaluator needs **one extra callback**
   [
   M_i[u]:=D_\beta B_i[u],\qquad B_i:=\frac{\partial H}{\partial \theta_i}\Big|_{\beta\ \mathrm{fixed}}.
   ]
3. Everything else fits into one loop over coordinates. This is very much in the spirit of Wood’s general framework: new model classes reduce to supplying standard likelihood derivatives, while the outer LAML machinery stays generic. ([School of Mathematics][1])

---

## 1. Unified calculus for (\theta=(\rho,\psi))

Write
[
F(\beta,\theta)=-\ell(\beta,\theta)+\frac12\beta^\top S(\theta)\beta,\qquad
s(\beta,\theta):=\nabla_\beta F,\qquad
H(\beta,\theta):=\nabla_\beta^2F.
]

At the mode (\hat\beta(\theta)), (s(\hat\beta,\theta)=0).

### 1.1 Coordinate objects the evaluator should consume

For each outer coordinate (\theta_i), define the **fixed-(\beta)** objects
[
a_i:=\partial_i F\big|*\beta,\qquad
g_i:=\partial_i s\big|*\beta,\qquad
B_i:=\partial_i H\big|*\beta,\qquad
\ell^S_i:=\partial_i \log|S|*+.
]

For each pair ((i,j)), define
[
a_{ij}:=\partial_{ij}^2F\big|*\beta,\qquad
g*{ij}:=\partial_{ij}^2 s\big|*\beta,\qquad
B*{ij}:=\partial_{ij}^2 H\big|*\beta,\qquad
\ell^S*{ij}:=\partial_{ij}^2\log|S|_+.
]

Now isolate the (\beta)-dependence of the likelihood Hessian with
[
C[u]:=D_\beta H_L[u],\qquad
Q[u,v]:=D_\beta^2 H_L[u,v].
]

And add the one new object needed for exact (\psi)-Hessians:
[
M_i[u]:=D_\beta B_i[u].
]

For pure (\rho), (B_i=A_i) is (\beta)-independent, so (M_i\equiv 0).

---

## 2. Master formulas

### 2.1 First-order mode response

[
\beta_i:=\frac{\partial \hat\beta}{\partial \theta_i}=-H^{-1}g_i.
]

### 2.2 First-order total Hessian drift

[
\dot H_i = B_i + C[\beta_i].
]

### 2.3 Profiled gradient

[
V_i:=\frac{\partial V}{\partial \theta_i}
= a_i+\frac12\operatorname{tr}(H^{-1}\dot H_i)-\frac12\ell^S_i.
]

This is the exact unified gradient formula.

So **Q1** has answer **yes**: the evaluator can be generalized by replacing (A_k) with the tuple ((a_i,g_i,B_i,\ell^S_i)). Nothing else is needed for the gradient.

---

## 3. Exact second-order formulas

### 3.1 Second mode response

Differentiate (H\beta_i+g_i=0):
[
H\beta_{ij} + \dot H_j,\beta_i + g_{ij} + B_i\beta_j = 0.
]

Using (\dot H_j=B_j+C[\beta_j]),
[
\boxed{
\beta_{ij}
==========

-H^{-1}\Big(g_{ij}+B_i\beta_j+B_j\beta_i+C[\beta_i]\beta_j\Big).
}
]

Because the third derivative tensor of a smooth scalar likelihood is symmetric,
[
C[\beta_i]\beta_j=C[\beta_j]\beta_i,
]
so numerically you can safely symmetrize this term as
[
\frac12\big(C[\beta_i]\beta_j + C[\beta_j]\beta_i\big).
]

### 3.2 Second total Hessian drift

[
\boxed{
\ddot H_{ij}
============

B_{ij}+C[\beta_{ij}]+M_i[\beta_j]+M_j[\beta_i]+Q[\beta_i,\beta_j].
}
]

### 3.3 Profiled Hessian

[
\boxed{
V_{ij}
======

\Big(a_{ij}-g_i^\top H^{-1}g_j\Big)
+
\frac12\Big[
\operatorname{tr}(H^{-1}\ddot H_{ij})
-------------------------------------

\operatorname{tr}(H^{-1}\dot H_j H^{-1}\dot H_i)
\Big]
-\frac12\ell^S_{ij}.
}
]

That is the exact joint Hessian for ((\rho,\psi)).

So the exact answer to **Q1/Q3/Q4/Q5** is:

* the **gradient** needs only ((a_i,g_i,B_i,\ell^S_i)) plus (C[\cdot]);
* the **Hessian** additionally needs ((a_{ij},g_{ij},B_{ij},\ell^S_{ij})), (Q[\cdot,\cdot]), and the new callback (M_i[\cdot]).

The current `correction/second_correction` interface is therefore **gradient-complete** but **not Hessian-complete** for general (\psi).

---

## 4. Specialization to (\rho) and (\psi)

### 4.1 (\rho_k)-coordinates

Let
[
A_k(\psi):=\lambda_k S_k(\psi),\qquad \lambda_k=e^{\rho_k}.
]

Then
[
a_{\rho_k}=\frac12\beta^\top A_k\beta,\qquad
g_{\rho_k}=A_k\beta,\qquad
B_{\rho_k}=A_k.
]

Second derivatives:
[
a_{\rho_k\rho_l}=
\begin{cases}
\frac12\beta^\top A_k\beta,&k=l,\
0,&k\neq l,
\end{cases}
\qquad
g_{\rho_k\rho_l}=
\begin{cases}
A_k\beta,&k=l,\
0,&k\neq l,
\end{cases}
\qquad
B_{\rho_k\rho_l}=
\begin{cases}
A_k,&k=l,\
0,&k\neq l.
\end{cases}
]

And (M_{\rho_k}\equiv 0).

### 4.2 (\psi_j)-coordinates

Let
[
S_j:=\partial_{\psi_j}S,\qquad
S_{ij}:=\partial^2_{\psi_i\psi_j}S.
]

Define family likelihood objects
[
a^{\ell}*j:=\partial*{\psi_j}(-\ell)\big|*\beta,\qquad
q_j:=\partial*{\psi_j}\nabla_\beta(-\ell)\big|*\beta,\qquad
L_j:=\partial*{\psi_j}H_L\big|*\beta,
]
and pairwise
[
a^{\ell}*{ij}:=\partial^2_{\psi_i\psi_j}(-\ell)\big|*\beta,\qquad
q*{ij}:=\partial^2_{\psi_i\psi_j}\nabla_\beta(-\ell)\big|*\beta,\qquad
L*{ij}:=\partial^2_{\psi_i\psi_j}H_L\big|_\beta.
]

Then
[
a_{\psi_j}=a^\ell_j+\frac12\beta^\top S_j\beta,\qquad
g_{\psi_j}=q_j+S_j\beta,\qquad
B_{\psi_j}=L_j+S_j,
]
and
[
a_{\psi_i\psi_j}=a^\ell_{ij}+\frac12\beta^\top S_{ij}\beta,\qquad
g_{\psi_i\psi_j}=q_{ij}+S_{ij}\beta,\qquad
B_{\psi_i\psi_j}=L_{ij}+S_{ij}.
]

Now
[
M_{\psi_j}[u]=D_\beta L_j[u].
]

This is the exact extra third-order mixed derivative you were worried about in **Q4**.

---

## 5. Explicit ((\rho_k,\psi_j)) cross-Hessian

Let
[
A_{k,j}:=\partial_{\psi_j}A_k
=\lambda_k,\partial_{\psi_j}S_k.
]

Then the fixed-(\beta) cross objects are
[
a_{k j}=\frac12\beta^\top A_{k,j}\beta,\qquad
g_{k j}=A_{k,j}\beta,\qquad
B_{k j}=A_{k,j}.
]

Also write
[
g_k=A_k\beta,\qquad
g_j=q_j+S_j\beta,\qquad
B_k=A_k,\qquad
B_j=L_j+S_j.
]

Then
[
\beta_k=-H^{-1}A_k\beta,\qquad
\beta_j=-H^{-1}(q_j+S_j\beta),
]
and
[
\boxed{
\beta_{k j}
===========

-H^{-1}\Big(
A_{k,j}\beta + A_k\beta_j + (L_j+S_j)\beta_k + C[\beta_k]\beta_j
\Big).
}
]

Since (M_{\rho_k}=0),
[
\boxed{
\ddot H_{k j}
=============

A_{k,j}+C[\beta_{k j}] + D_\beta L_j[\beta_k] + Q[\beta_k,\beta_j].
}
]

So the mixed outer Hessian entry is
[
\boxed{
V_{k j}
=======

\left(\frac12\beta^\top A_{k,j}\beta - g_k^\top H^{-1}g_j\right)
+
\frac12\left[
\operatorname{tr}(H^{-1}\ddot H_{k j})
--------------------------------------

\operatorname{tr}(H^{-1}\dot H_j H^{-1}\dot H_k)
\right]
-\frac12\ell^S_{k j},
}
]
with
[
\dot H_k=A_k+C[\beta_k],\qquad
\dot H_j=L_j+S_j+C[\beta_j].
]

This answers **Q3**. The split is:

* **penalty-only / generic**:
  [
  A_k,\ A_{k,j},\ S_j,\ \ell^S_{k j};
  ]
* **family-specific**:
  [
  a^\ell_j,\ q_j,\ L_j,\ q_{ij},\ L_{ij},\ D_\beta L_j[\cdot],\ C[\cdot],\ Q[\cdot,\cdot].
  ]

So the cross derivative fits the same generic pattern, but the **existing** `second_correction` is not enough unless you add (D_\beta L_j[\cdot]).

---

## 6. What “relinearization” means at (\psi\neq 0)

If
[
X(\psi)=X_0+\sum_j \psi_j X_{\tau_j}+\frac12\sum_{i,j}\psi_i\psi_j X_{ij},
]
then at current (\psi),
[
\partial_{\psi_j}X(\psi)=X_{\tau_j}+\sum_i \psi_i X_{ij}=: \tilde X_{\tau_j}.
]

So the evaluator must use the **current derivative**
[
\tilde X_{\tau_j}=\partial_{\psi_j}X(\psi),
]
not the base-point derivative (X_{\tau_j}), unless (\psi=0).

Exactly the same applies to the penalty:
[
\partial_{\psi_j}S(\rho,\psi)
=============================

\sum_k \lambda_k\left(S_{k,\tau_j}+\sum_i \psi_i S_{k,ij}\right).
]

So **Q2** answer:

* use the **relinearized current derivatives** in all fixed-(\beta) objects (a_j,g_j,B_j);
* use the corresponding current (\partial_{\psi_j}S) and (\partial_{\psi_i\psi_j}S) in (\ell^S_j,\ell^S_{ij});
* mode response uses these current derivatives automatically through (g_j).

Wood explicitly notes that when the model matrix depends on smoothing/hyperparameters, the derivative expressions become more complicated but the same outer-optimization logic still applies. ([University of Bath Personal Homepages][2])

---

## 7. When does (M_j[u]=D_\beta B_j[u]) vanish?

This is **Q4(a)**.

It vanishes whenever (B_j) is (\beta)-independent.

That includes:

* pure (\rho) directions;
* Gaussian identity with design motion but fixed observation weights:
  [
  H_L=X^\top W X,\quad W\ \text{constant},
  ]
  so
  [
  B_j=\dot X_j^\top W X + X^\top W \dot X_j,
  ]
  which is independent of (\beta).

It does **not** generally vanish for non-Gaussian models with moving design, because then (W=W(\eta)) and
[
\dot\eta_j = \dot X_j \beta
]
feeds (\beta) into (B_j).

So for logistic, Poisson-log, Gamma-log, NB-log, Beta-logit, etc., (M_j) is usually nonzero if (\psi) moves (X).

### 7.1 Link-wiggle form

If
[
H_L = J^\top W J,
]
with (J=J(\beta,\psi)), then for an outer coordinate (j),
[
B_j = J_j^\top WJ + J^\top WJ_j + J^\top \operatorname{diag}(W' \odot \dot\eta_j)J,
]
where
[
J_j:=\partial_{\psi_j}J\big|*\beta,\qquad
\dot\eta_j:=\partial*{\psi_j}\eta\big|_\beta.
]

Then
[
M_j[u]=D_\beta B_j[u]
]
is expressible using (W'), (W''), (D_\beta J[u]), (D_\beta J_j[u]), and (D_\beta\dot\eta_j[u]). So **Q4(b)** answer is: yes, it can be written in terms of the same weight-derivative machinery, but **not** from (W',W'') alone; you also need the Jacobian-sensitivity pieces.

### 7.2 Can it be dropped?

For **Q4(c)**:

* dropping (M_j[u]) does **not** affect the gradient;
* it gives an **approximate outer Hessian**;
* that is usually fine for BFGS, trust-region, or ARC;
* it is **not** fine if you want an exact Newton Hessian and quadratic convergence.

So: safe to drop for a robust quasi-Newton implementation, unsafe to claim exact Newton.

---

## 8. Minimal family API for (\psi)

For the exact gradient and Hessian, the minimal family contribution is:

Per coordinate (i\in\psi):
[
a_i^\ell,\ q_i,\ L_i.
]

Per pair (i,j\in\psi):
[
a_{ij}^\ell,\ q_{ij},\ L_{ij}.
]

And curvature oracles:
[
C[u]=D_\beta H_L[u],\qquad
Q[u,v]=D_\beta^2H_L[u,v],\qquad
M_i[u]=D_\beta L_i[u].
]

Add the penalty derivatives
[
S_i,\ S_{ij},\ \partial_{\psi_j}S_k,
]
and you are complete.

So **Q5** answer is:

* your list 1–6 is almost complete;
* the missing piece for the **exact outer Hessian** is
  [
  D_\beta\left(\frac{\partial H_L}{\partial \psi_i}\Big|_\beta\right)[u].
  ]
  That is exactly (M_i[u]).

If you provide (B_i) and (B_{ij}) directly, then (M_i) is still needed unless (B_i) is known to be (\beta)-independent.

---

## 9. Profiled Gaussian dispersion under (\psi)

Let
[
\nu:=n-M_p,\qquad
\hat\phi(\theta)=\frac{D_p(\hat\beta(\theta),\theta)}{\nu},\qquad
D_p=-2\ell+\beta^\top S\beta.
]

Then
[
V(\theta)=\frac{\nu}{2}\log\hat\phi(\theta)+\frac12\log|H|*+ - \frac12\log|S|*+ + c.
]

At the mode,
[
\partial_i D_p = 2 a_i
]
because (D_p=2F) up to constants and the envelope term vanishes.

Hence
[
\boxed{
\partial_i V
============

\frac{\nu}{2}\frac{\partial_i D_p}{D_p}
+
\frac12\operatorname{tr}(H^{-1}\dot H_i)
-\frac12\ell^S_i
================

\frac{a_i}{\hat\phi}
+
\frac12\operatorname{tr}(H^{-1}\dot H_i)
-\frac12\ell^S_i.
}
]

So for (\psi_j),
[
\boxed{
\partial_{\psi_j}V
==================

\frac{a_{\psi_j}}{\hat\phi}
+
\frac12\operatorname{tr}(H^{-1}\dot H_{\psi_j})
-\frac12\ell^S_{\psi_j}.
}
]

That is the full answer to **Q6**: profiling (\phi) adds no new mode-response terms; it only rescales the fixed-(\beta) part by (1/\hat\phi).

---

## 10. Implementation-ready unified evaluator (D1)

Use the notation above. Then:

```text
Input:
  beta_hat, factorization of H
  for each coordinate i:
      a_i, g_i, B_i, ldS_i
  for each pair (i,j):
      a_ij, g_ij, B_ij, ldS_ij
  provider:
      C(u)   = D_beta H_L[u]
      Q(u,v) = D_beta^2 H_L[u,v]
      M_i(u) = D_beta B_i[u]   # zero for beta-independent B_i

Step 1: first-order responses
  for i = 1..q:
      beta_i = - solve(H, g_i)
      H1_i   = B_i + C(beta_i)
      grad_i = a_i + 0.5 * tr(H^{-1} H1_i) - 0.5 * ldS_i

Step 2: second-order responses
  for i <= j:
      rhs_ij =
          g_ij
        + B_i * beta_j
        + B_j * beta_i
        + 0.5 * ( C(beta_i) * beta_j + C(beta_j) * beta_i )

      beta_ij = - solve(H, rhs_ij)

      H2_ij =
          B_ij
        + C(beta_ij)
        + M_i(beta_j)
        + M_j(beta_i)
        + Q(beta_i, beta_j)

      hess_ij =
          a_ij
        - g_i^T solve(H, g_j)
        + 0.5 * tr(H^{-1} H2_ij)
        - 0.5 * tr(H^{-1} H1_j H^{-1} H1_i)
        - 0.5 * ldS_ij

      set hess_ji = hess_ij
```

### Which terms need the new callback?

Only
[
M_i(\beta_j)+M_j(\beta_i).
]

Everything else is covered by the current correction logic, provided you reinterpret it as

* `correction(direction)` = (C[\text{direction}]),
* `second_correction(dir1, dir2, dir12)` = (C[\text{dir12}] + Q[\text{dir1},\text{dir2}]).

So the exact extension is:

* keep `correction`;
* generalize `second_correction` or add a separate `beta_dep_fixed_drift(i, direction)` for (M_i[\cdot]).

---

## 11. Sharing likelihood between REML and HMC

The right split is a **two-tier likelihood API**.

### Tier A: fast score oracle for HMC

At a given coefficient vector (\beta), return
[
\ell(\beta),\qquad
\nabla_\beta \ell(\beta).
]

This is all NUTS needs for the conditional posterior
[
\log p(\beta\mid y,\lambda)=\ell(\beta)-\frac12\beta^\top S\beta.
]

That path should have no curvature overhead.

### Tier B: curvature extension for REML/LAML

On top of Tier A, optionally provide
[
H_L,\quad D_\beta H_L[u],\quad D_\beta^2 H_L[u,v],
]
and the fixed-(\beta) hyper-derivatives
[
a_i^\ell,\ q_i,\ L_i,\ a_{ij}^\ell,\ q_{ij},\ L_{ij},\ M_i[u].
]

That is exactly how Wood’s “general smooth model” framework is organized conceptually: a generic outer smoother-selection layer above a model-specific derivative layer. ([School of Mathematics][1])

So the clean abstraction is:

* **base trait**: log-likelihood + score;
* **extended trait**: curvature and hyper-derivatives.

That gives HMC the fast path and REML the full path.

---

## 12. Inner vs outer hyperparameters

This answers the link-wiggle / spatial-anisotropy question.

Mathematically, “inner” vs “outer” is not a different calculus. It is a modeling choice.

A parameter should be **inner** if you want it in the joint mode vector and are willing to:

* solve for it in the inner Newton/PIRLS system,
* include its Hessian block in (H),
* integrate it out by the Laplace approximation or sample it in HMC.

A parameter should be **outer** if you want to profile it, usually because:

* its dimension is tiny,
* it changes the basis/design/penalty structure nonlinearly,
* it is not naturally given a quadratic prior,
* or its inclusion in the inner system would make the local quadratic approximation poor.

So:

* link-wiggle coefficients (\theta) are **inner** because they are just extra coefficients with a quadratic penalty block (\lambda_\theta S_\theta);
* anisotropy parameters (\psi) are **outer** because they move (X) and (S) structurally.

Both use the **same** (D_\beta H_L) machinery. The difference is only whether the parameter lives in (\alpha) or in (\theta).

---

# 13. Family derivative reference (D2–D4)

Let (\omega_i) be the prior/observation weight for observation (i). Below I drop the subscript (i).

I use:

* (\ell(\eta)): log-likelihood contribution;
* (s(\eta)=\partial \ell/\partial \eta);
* (w(\eta)=-\partial^2\ell/\partial \eta^2);
* (c(\eta)=\partial w/\partial \eta);
* (d(\eta)=\partial^2 w/\partial \eta^2).

For HMC, the posterior gradient is always
[
\nabla_\beta\log p(\beta\mid y,\lambda)=X^\top r - S\beta,
]
with the appropriate residual (r).

---

## 13.1 Poisson, log link

[
\mu=e^\eta.
]

[
\ell=\omega\big(y\eta-\mu-\log(y!)\big),\qquad
s=\omega(y-\mu).
]

[
w=\omega\mu,\qquad
c=\omega\mu,\qquad
d=\omega\mu.
]

Observed = Fisher. Canonical link: **yes**.

HMC residual:
[
r=\omega(y-\mu),\qquad
\nabla_\beta \ell = X^\top r.
]

---

## 13.2 Poisson, identity link

[
\mu=\eta,\qquad \eta>0.
]

[
\ell=\omega\big(y\log\eta-\eta-\log(y!)\big),\qquad
s=\omega\left(\frac{y}{\eta}-1\right).
]

Observed information:
[
w=\omega\frac{y}{\eta^2},\qquad
c=-2\omega\frac{y}{\eta^3},\qquad
d=6\omega\frac{y}{\eta^4}.
]

Fisher information:
[
w_F=\omega\frac1\eta,\qquad
c_F=-\omega\frac1{\eta^2},\qquad
d_F=2\omega\frac1{\eta^3}.
]

Observed = Fisher: **no**. Canonical link: **no**.

HMC residual:
[
r=\omega\left(\frac{y}{\eta}-1\right),\qquad
\nabla_\beta \ell = X^\top r.
]

---

## 13.3 Gamma, log link

Shape (\alpha), rate (\alpha/\mu), mean (\mu=e^\eta).

[
\ell
====

\omega\Big(
\alpha\log\alpha-\log\Gamma(\alpha)+(\alpha-1)\log y-\alpha\eta-\alpha y e^{-\eta}
\Big).
]

[
s=\omega\alpha\left(\frac{y}{\mu}-1\right).
]

Observed information:
[
w=\omega\alpha\frac{y}{\mu},\qquad
c=-\omega\alpha\frac{y}{\mu},\qquad
d=\omega\alpha\frac{y}{\mu}.
]

Fisher information:
[
w_F=\omega\alpha,\qquad c_F=0,\qquad d_F=0.
]

Observed = Fisher: **no**. Canonical link: **no**.

HMC residual:
[
r=\omega\alpha\left(\frac{y}{\mu}-1\right),\qquad
\nabla_\beta \ell = X^\top r.
]

---

## 13.4 Gamma, inverse link

[
\mu=\frac1\eta,\qquad \eta>0.
]

[
\ell
====

\omega\Big(
\alpha\log\alpha-\log\Gamma(\alpha)+(\alpha-1)\log y+\alpha\log\eta-\alpha y\eta
\Big).
]

[
s=\omega\alpha\left(\frac1\eta-y\right).
]

[
w=\omega\alpha\frac1{\eta^2},\qquad
c=-2\omega\alpha\frac1{\eta^3},\qquad
d=6\omega\alpha\frac1{\eta^4}.
]

Observed = Fisher: **yes**. Canonical link: **yes** up to the usual sign convention for the Gamma canonical parameter.

HMC residual:
[
r=\omega\alpha\left(\frac1\eta-y\right),\qquad
\nabla_\beta \ell = X^\top r.
]

---

## 13.5 Negative binomial, log link

Take size (r), mean (\mu=e^\eta), variance (\mu+\mu^2/r).

[
\ell
====

\omega\Big(
\log\Gamma(y+r)-\log\Gamma(r)-\log\Gamma(y+1)
+y\eta-(y+r)\log(r+\mu)+r\log r
\Big).
]

[
s=\omega,\frac{r(y-\mu)}{r+\mu}.
]

Observed information:
[
w=\omega,\frac{r\mu(r+y)}{(r+\mu)^2},
]
[
c=\omega,\frac{r\mu(r+y)(r-\mu)}{(r+\mu)^3},
]
[
d=\omega,\frac{r\mu(r+y)(r^2-4r\mu+\mu^2)}{(r+\mu)^4}.
]

Fisher information:
[
w_F=\omega,\frac{r\mu}{r+\mu},
]
[
c_F=\omega,\frac{r^2\mu}{(r+\mu)^2},
]
[
d_F=\omega,\frac{r^2\mu(r-\mu)}{(r+\mu)^3}.
]

Observed = Fisher: **no**. Canonical link: **no** for NB2-log.

HMC residual:
[
r=\omega,\frac{r(y-\mu)}{r+\mu},\qquad
\nabla_\beta \ell = X^\top r.
]

---

## 13.6 Beta, logit link

[
\mu=\operatorname{logit}^{-1}(\eta),\qquad
m=\mu(1-\mu),\qquad
t=1-2\mu,
]
with precision (\phi).

[
\ell
====

\omega\Big(
\log\Gamma(\phi)-\log\Gamma(\mu\phi)-\log\Gamma((1-\mu)\phi)
+(\mu\phi-1)\log y + ((1-\mu)\phi-1)\log(1-y)
\Big).
]

Define
[
A=\log\frac{y}{1-y}-\psi_0(\mu\phi)+\psi_0((1-\mu)\phi),
]
[
B=\psi_1(\mu\phi)+\psi_1((1-\mu)\phi),
]
[
C=\psi_2(\mu\phi)-\psi_2((1-\mu)\phi),
]
[
D=\psi_3(\mu\phi)+\psi_3((1-\mu)\phi),
]
where (\psi_r) is the order-(r) polygamma.

Then
[
s=\omega \phi m A.
]

Observed information:
[
w=\omega\phi\big(\phi m^2 B - m t A\big),
]
[
c=\omega\phi\Big(-m(t^2-2m)A + 3\phi m^2 t B + \phi^2 m^3 C\Big),
]
[
d=\omega\phi\Big(
-mt(t^2-8m)A
+\phi m^2(7t^2-8m)B
+6\phi^2 m^3 t C
+\phi^3 m^4 D
\Big).
]

Fisher information:
[
w_F=\omega\phi^2 m^2 B,
]
[
c_F=\omega\phi^2\Big(2m^2 t B + \phi m^3 C\Big),
]
[
d_F=\omega\phi^2\Big(4m^2(t^2-m)B + 5\phi m^3 t C + \phi^2 m^4 D\Big).
]

Observed = Fisher: **no**.

HMC residual:
[
r=\omega\phi m A,\qquad
\nabla_\beta \ell = X^\top r.
]

---

## 13.7 Tweedie, log link

[
\mu=e^\eta,\qquad V(\mu)=\mu^p.
]

The exact exponential-dispersion form is
[
\ell
====

\omega\left[
\frac{1}{\phi}
\left(
\frac{y\mu^{1-p}}{1-p}
----------------------

\frac{\mu^{2-p}}{2-p}
\right)
+
c(y,\phi,p)
\right],
]
where (c(y,\phi,p)) is the Tweedie base-measure term. For generic (p) that term is given by the usual Tweedie series rather than an elementary closed form.

[
s=\omega,\frac{y-\mu}{\phi,\mu^{p-1}}.
]

Observed information:
[
w=\omega,\frac{(2-p)\mu^{2-p} + (p-1)y\mu^{1-p}}{\phi},
]
[
c=\omega,\frac{(2-p)^2\mu^{2-p} - (p-1)^2 y\mu^{1-p}}{\phi},
]
[
d=\omega,\frac{(2-p)^3\mu^{2-p} + (p-1)^3 y\mu^{1-p}}{\phi}.
]

Fisher information:
[
w_F=\omega,\frac{\mu^{2-p}}{\phi},\qquad
c_F=\omega,\frac{(2-p)\mu^{2-p}}{\phi},\qquad
d_F=\omega,\frac{(2-p)^2\mu^{2-p}}{\phi}.
]

Observed = Fisher: **no** except special subcases.

HMC residual:
[
r=\omega,\frac{y-\mu}{\phi,\mu^{p-1}},\qquad
\nabla_\beta \ell = X^\top r.
]

---

# 14. Correction formula (D_\beta H_L[u]) (D3)

For a single-predictor model with fixed (X) and scalar per-observation Hessian weight (w_i(\eta_i)),
[
H_L = X^\top \operatorname{diag}(w) X,
]
so
[
\boxed{
D_\beta H_L[u] = X^\top \operatorname{diag}\big(c \odot (Xu)\big)X,
}
]
with (c_i=\partial w_i/\partial\eta_i).

That is true for both observed and Fisher information, provided (w) is the weight you actually use.

So for the families above, the correction formula is exactly the same with the corresponding (c) or (c_F).

### General noncanonical exponential-dispersion form

Let (h(\eta)=g^{-1}(\eta)), (\mu=h(\eta)), and write
[
h_1=h'(\eta),\quad h_2=h''(\eta),\quad h_3=h'''(\eta),
]
[
V=V(\mu),\quad V_1=V'(\mu),\quad V_2=V''(\mu).
]

Then
[
\partial_\eta \ell = \frac{(y-\mu)h_1}{\phi V}.
]

Fisher weight:
[
w_F = \frac{h_1^2}{\phi V},
]
[
c_F = \frac{2 h_1 h_2 V - h_1^3 V_1}{\phi V^2}.
]

Observed weight:
[
w_{\rm obs}
===========

## \frac{h_1^2}{\phi V}

(y-\mu),
\frac{h_2 V - h_1^2 V_1}{\phi V^2}.
]

Define
[
B:=\frac{h_2 V - h_1^2 V_1}{\phi V^2}.
]
Then
[
c_{\rm obs}=c_F + h_1 B - (y-\mu) B_\eta,
]
with
[
B_\eta
======

\frac{
h_3 V^2
-3 h_1 h_2 V V_1
-h_1^3 V V_2
+2 h_1^3 V_1^2
}{\phi V^3}.
]

So the exact observed-information correction for noncanonical links needs (V''). That is the correction to your D3(b): (V) and (V') are enough for **Fisher**, but not for the exact **observed** correction.

---

# 15. NUTS posterior gradients (D4)

These are the HMC gradients before subtracting (S\beta).

For all single-predictor cases above:
[
\nabla_\beta \ell = X^\top r,
]
with residual (r) as listed.

So the posterior gradient is
[
\boxed{
\nabla_\beta\log p(\beta\mid y,\lambda)=X^\top r - S\beta.
}
]

### 15.1 Survival (Royston–Parmar, single predictor)

Given
[
\ell_i = \delta_i\big(\eta_{1i}+\log s_i\big)-e^{\eta_{1i}}+e^{\eta_{0i}},
\qquad s_i=d_i^\top\beta,
]
the contribution is
[
\nabla_\beta \ell_i
===================

\delta_i x_{1i}
+\delta_i \frac{d_i}{s_i}
-------------------------

e^{\eta_{1i}}x_{1i}
+
e^{\eta_{0i}}x_{0i}.
]

Summing:
[
\boxed{
\nabla_\beta \ell
=================

X_1^\top(\delta-e^{\eta_1})
+
X_0^\top e^{\eta_0}
+
D^\top(\delta \oslash s),
}
]
where (\oslash) is elementwise division.

This is not a single (X^\top r) with the original (X), but it is an augmented sparse linear form.

### 15.2 Gaussian location-scale

If the scale predictor is on (\sigma) directly,
[
\ell_i = -\log\sigma_i - \frac{(y_i-\mu_i)^2}{2\sigma_i^2} + c.
]

Then
[
\nabla_{\beta_\mu}\ell = X_\mu^\top\left(\frac{y-\mu}{\sigma^2}\right),
]
[
\nabla_{\beta_\sigma}\ell = X_\sigma^\top\left(\frac{r^2-1}{\sigma}\right),
\qquad r=\frac{y-\mu}{\sigma}.
]

If the scale predictor is (\eta_\sigma=\log\sigma), then simply multiply the direct-(\sigma) score by (\sigma):
[
\nabla_{\beta_{\log\sigma}}\ell = X_\sigma^\top(r^2-1).
]

---

# 16. Multi-block questions: Firth, GAMLSS curvature, ALO, covariance

## 16.1 Firth bias reduction (Q7)

For
[
J(\theta)=\frac12\log|I(\hat\alpha(\theta),\theta)|,
]
the exact gradient is
[
\boxed{
J_i
===

\frac12 \operatorname{tr}\Big(I^{-1}\dot I_i\Big),
\qquad
\dot I_i = I_i^{\rm fix}+D_\alpha I[\alpha_i].
}
]

For pure (\rho_k), if (I) is the likelihood/Fisher information only, then (I_k^{\rm fix}=0), so
[
J_k = \frac12 \operatorname{tr}\big(I^{-1} D_\alpha I[\alpha_k]\big),
\qquad
\alpha_k=-H^{-1}g_k.
]

The exact Hessian mirrors the LAML Hessian:
[
J_{ij}
======

\frac12\Big[
\operatorname{tr}(I^{-1}\ddot I_{ij})
-------------------------------------

\operatorname{tr}(I^{-1}\dot I_j I^{-1}\dot I_i)
\Big],
]
with
[
\ddot I_{ij}
============

I_{ij}^{\rm fix}
+D_\alpha I[\alpha_{ij}]
+D_\alpha I_i^{\rm fix}[\alpha_j]
+D_\alpha I_j^{\rm fix}[\alpha_i]
+D_\alpha^2 I[\alpha_i,\alpha_j].
]

If (I) is block diagonal, the trace decomposes blockwise. If (I) is coupled, you need the full joint (I^{-1}) contraction. Firth’s canonical-exponential-family/Jeffreys-prior connection is the classical result of Firth (1993). ([OUP Academic][3])

## 16.2 Gaussian location-scale cross-curvature (Q9)

Using the user’s direct-(\sigma) parameterization,
[
H_{\mu\mu}=X_\mu^\top \operatorname{diag}(\sigma^{-2}) X_\mu,
]
[
H_{\mu\sigma}=H_{\sigma\mu}
===========================

X_\mu^\top \operatorname{diag}(2r,\sigma^{-2}) X_\sigma,
]
[
H_{\sigma\sigma}
================

X_\sigma^\top \operatorname{diag}((3r^2-1)\sigma^{-2}) X_\sigma.
]

For a perturbation (u=(u_\mu,u_\sigma)), let
[
\delta\mu=X_\mu u_\mu,\qquad
\delta\sigma=X_\sigma u_\sigma.
]

Then
[
D_\beta H_L[u]
==============

\begin{pmatrix}
X_\mu^\top \operatorname{diag}(q_{\mu\mu}) X_\mu &
X_\mu^\top \operatorname{diag}(q_{\mu\sigma}) X_\sigma[1mm]
X_\sigma^\top \operatorname{diag}(q_{\sigma\mu}) X_\mu &
X_\sigma^\top \operatorname{diag}(q_{\sigma\sigma}) X_\sigma
\end{pmatrix},
]
with
[
q_{\mu\mu}= -2,\delta\sigma,\sigma^{-3},
]
[
q_{\mu\sigma}=q_{\sigma\mu}
= -(2\delta\mu + 6r,\delta\sigma)\sigma^{-3},
]
[
q_{\sigma\sigma}
================

\big(-6r,\delta\mu + (2-12r^2)\delta\sigma\big)\sigma^{-3}.
]

So the answer to **Q9(b)** is yes: it decomposes into within-block and cross-block (X_a^\top\operatorname{diag}(\cdot)X_b) terms.

For **Q9(c)**, with (B) predictors, each Hessian block weight (w_{ab}) depends on all (B) predictors. Then (D H[u]) needs (B^2) first-order weight vectors
[
c_{ab}(u)=\sum_c \frac{\partial w_{ab}}{\partial \eta_c},\delta\eta_c,
]
and (D^2H[u,v]) needs
[
d_{ab}(u,v)=\sum_{c,d}\frac{\partial^2 w_{ab}}{\partial \eta_c\partial \eta_d},\delta\eta_c^{(u)}\delta\eta_d^{(v)}.
]
Naively that is (B^2\times B(B+1)/2) second-order weight-like vectors, reduced by symmetry if present.

## 16.3 ALO for multi-predictor models (D5)

Let (\eta_i\in\mathbb R^B) be the vector of predictors for observation (i), (W_i\in\mathbb R^{B\times B}) the per-observation Hessian block of the NLL, and (X_i) the (B\times p_{\rm tot}) observation Jacobian row block. Then
[
H = \sum_i X_i^\top W_i X_i + S,
]
and the joint hat matrix is
[
\mathcal H = \mathcal X H^{-1}\mathcal X^\top \mathcal W.
]

The leverage for observation (i) is the (B\times B) block
[
\mathcal H_{ii}=X_i H^{-1} X_i^\top W_i.
]

So (h_i) is a **matrix**, not a scalar. Its trace is the EDF contribution.

If (s_i=\nabla_{\eta_i}\text{NLL}*i(\hat\eta_i)), then the one-step ALO correction is
[
\boxed{
\tilde\eta_i^{(-i)}
\approx
\hat\eta_i
+
(I_B-\mathcal H*{ii})^{-1}\mathcal H_{ii} W_i^{-1}s_i.
}
]

Equivalently, in score form (u_i=-s_i),
[
\tilde\eta_i^{(-i)}
\approx
\hat\eta_i
----------

(I_B-\mathcal H_{ii})^{-1}\mathcal H_{ii} W_i^{-1}u_i.
]

So the correction is **joint**, not per-predictor separately.

Then the approximate LOO log-likelihood is
[
\boxed{
\mathrm{ALO}=\sum_i \ell_i\big(\tilde\eta_i^{(-i)}\big).
}
]

This is the multi-predictor extension of the usual quadratic/influence ALO construction. ([Massachusetts Institute of Technology][4])

## 16.4 Corrected covariance (V_\alpha^*) (Q10, D6)

Let
[
J_\alpha := \frac{\partial \hat\alpha}{\partial \theta}
=======================================================

[\alpha_1,\ldots,\alpha_q],\qquad
\alpha_i=-H^{-1}g_i.
]

Then the first-order smoothing/hyperparameter uncertainty correction is
[
\boxed{
V_\alpha^* \approx H^{-1} + J_\alpha,V_\theta,J_\alpha^\top,
\qquad
V_\theta=(\nabla_\theta^2 V)^{-1}.
}
]

This must be computed **jointly** because (H) is joint and the columns (\alpha_i) come from the joint solve.

The (\psi)-columns are already computed in the LAML gradient step:
[
\alpha_j^{(\psi)}=-H^{-1}g_j^{(\psi)}.
]
So they are essentially free once you have the unified evaluator.

For block (b), with selector (E_b),
[
\boxed{
V_{\beta_b}^* = E_b^\top V_\alpha^* E_b.
}
]
Cross-block coupling contributes through the full (H^{-1}) and through the full (V_\theta).

---

# 17. Smoothness of (\log|S(\theta)|_+), reparameterization, indefiniteness

## 17.1 Smoothness of (\log|S|_+) (Q8)

(\log|S(\theta)|_+) is (C^2) on any domain where:

1. the rank of (S(\theta)) is constant, and
2. the positive eigenvalues stay bounded away from (0).

A sufficient condition is: there is a **fixed nullspace** (N) for all (\theta), and on the orthogonal complement (N^\perp),
[
\bar S(\theta):=Q^\top S(\theta)Q
]
is (C^2) and positive definite.

If rank changes, the pseudo-determinant is not smooth at the transition because (\log\lambda) blows up as (\lambda\downarrow 0).

Best remedy: **constrain the optimizer to a constant-rank manifold** or parameterize directly on the reduced positive subspace. Softening with (\log(\lambda+\varepsilon)) is a useful numerical regularization, but it changes the target and therefore biases REML.

If each (S_k(\psi)) has constant rank but their combination does not, then yes, this can create practical trouble. If all (S_k(\psi)) are PSD and share the same nullspace, then the combined rank is stable.

## 17.2 Reparameterization invariance (Q12)

For a (\theta)-independent nonsingular linear change of coefficient coordinates
[
\tilde X = X D^{-1},\qquad \tilde\beta = D\beta,\qquad \tilde S = D^{-T} S D^{-1},
]
the inner objective value at the mode is exactly invariant:
[
F(\hat\beta,\theta)=\tilde F(\hat{\tilde\beta},\theta).
]

Also
[
\tilde H = D^{-T} H D^{-1}.
]

So the REML/LAML objective is invariant **up to a (\theta)-independent additive constant** whenever the structural positive/null subspaces are fixed. Therefore (\nabla_\theta V) and (\nabla_\theta^2V) are invariant.

For full determinants the constant is exactly (-2\log|\det D|). For pseudo-determinants the exact constant is the log-determinant of the restriction of (D) to the positive subspace; the simple coordinate formula (-2\sum \log d_j) is only exact when (D) preserves the null/range splitting.

For the profiled Gaussian deviance
[
D_p = |y-X\hat\beta|^2 + \hat\beta^\top S\hat\beta,
]
invariance is exact:
[
y-X\hat\beta = y-\tilde X\hat{\tilde\beta},\qquad
\hat\beta^\top S\hat\beta=\hat{\tilde\beta}^\top \tilde S \hat{\tilde\beta}.
]

## 17.3 Indefinite (H) (Q13)

Using one operator for (\log|H|_+) and another for (H^{-1}) is mathematically inconsistent: the gradient is no longer the derivative of a single scalar objective.

The clean fix is to choose one smooth positive spectral regularization
[
\mathcal R_\varepsilon(H)=U,\mathrm{diag}(r_\varepsilon(\sigma_i)),U^\top,
]
with (r_\varepsilon(\sigma)>0) smooth, and use it **for both** inverse and logdet.

A good choice is
[
r_\varepsilon(\sigma)=\frac12\Big(\sigma+\sqrt{\sigma^2+4\varepsilon^2}\Big),
]
because it is (C^\infty), positive, and asymptotically equals (\sigma) for large positive (\sigma).

Then use
[
\log|\mathcal R_\varepsilon(H)|,\qquad \mathcal R_\varepsilon(H)^{-1}
]
everywhere.

Hard clamping (\max(\sigma,\varepsilon)) is not smooth. Absolute value is not differentiable at (0). Squaring to (H^\top H) changes the objective too much.

In practice, true indefiniteness is uncommon for convex canonical GLMs with fixed design and PSD penalties, but much more plausible in noncanonical families, location-scale/shape models, survival models, and very flexible likelihoods.

---

# 18. Active inequality constraints (Q14 and addendum)

For (\theta)-independent inequality constraints (A\beta\ge b), the constrained value-function envelope theorem still says the derivative of the **constrained optimum value** is the fixed-(\beta) derivative of the Lagrangian. The real issue is not an extra (\mu^\top A\beta_i) term; it is that the Laplace approximation is now on a **face** or in a **truncated Gaussian** neighborhood, so the usual unconstrained (\log|H|) term is not the whole story.

Practical rule:

* if the active set is fixed and strict complementarity holds, projecting into the free subspace gives the correct **within-face** curvature;
* the approximation breaks down near active-set changes, where the objective is only piecewise smooth.

A barrier version,
[
F_\tau(\beta,\theta)=F(\beta,\theta)-\tau\sum_j \log(\beta_j-b_j),
]
is smooth and yields a standard Laplace correction with barrier Hessian included in (H). This is the cleanest way to restore differentiability, but it changes the target problem.

For Firth on a constrained face, the mathematically consistent face approximation is
[
\frac12\log|P_{\mathcal F}^\top I P_{\mathcal F}|,
]
with (P_{\mathcal F}) spanning the free tangent space. That dimension jumps when the active set changes. Barrier regularization smooths that out. Keeping the full-space Firth penalty and only projecting its gradient is heuristic, not exact.

---

# 19. Convolutions, nonpolynomial hyperparameters, and exactness questions

## 19.1 Nonpolynomial hyperparameters (Q11)

The calculus does **not** require a polynomial dependence on (\psi). It only requires local derivatives at the current point:
[
\partial_\psi X,\ \partial_{\psi\psi}^2 X,\ \partial_\psi S,\ \partial_{\psi\psi}^2 S.
]

So yes: for Matérn (\kappa), SAS (\epsilon), etc., treat them as ordinary outer coordinates and compute the derivatives at the current value, analytically, by AD, or numerically.

That is just Newton on a smooth manifold of models. The quadratic model is local, so the usual trust-region/line-search safeguards are enough.

AD is usually the cleanest option when analytic derivatives are painful.

## 19.2 CLogLog Gaussian convolution (Q15a)

For CLogLog,
[
E[1-e^{-e^\eta}] = 1 - E[e^{-Y}],\qquad Y\sim \mathrm{Lognormal}(\mu,\sigma^2).
]

So the problem is the **Laplace transform of a lognormal** at (1). The literature on that transform uses analytic continuation, Mellin–Barnes integrals, and approximation schemes, which is a strong sign that there is no simple practical elementary closed form. ([arXiv][5])

My recommendation here is not low-order GHQ. Use either:

* a fixed high-order Hermite rule, or
* a Mellin–Barnes / contour-based routine if you want high smoothness.

## 19.3 Probit Gaussian convolution (Q15b)

[
f(\mu,\sigma)=E[\Phi(\eta)] = \Phi!\left(\frac{\mu}{\sqrt{1+\sigma^2}}\right).
]

Let
[
A=1+\sigma^2,\qquad q=\mu A^{-1/2}.
]

Then pure (\mu)-derivatives have the Hermite form
[
\partial_\mu^n f
================

A^{-n/2}(-1)^{n-1}H_{n-1}(q),\phi(q),\qquad n\ge 1,
]
where (H_k) are the probabilists’ Hermite polynomials.

So explicitly:
[
f_\mu = A^{-1/2}\phi(q),
]
[
f_{\mu\mu} = -A^{-1} q,\phi(q),
]
[
f_{\mu\mu\mu} = A^{-3/2}(q^2-1)\phi(q),
]
[
f_{\mu\mu\mu\mu} = A^{-2}(3q-q^3)\phi(q).
]

For (\sigma),
[
f_\sigma = -\mu\sigma A^{-3/2}\phi(q),
]
[
f_{\sigma\sigma}
================

-\mu A^{-7/2}\big(\mu^2\sigma^2 - 3\sigma^2A + A^2\big)\phi(q).
]

Higher (\sigma)- and mixed derivatives are still closed-form polynomials in ((\mu,\sigma)) times (\phi(q)), so probit is completely tractable.

## 19.4 SAS link (Q15c,d)

For SAS there is no comparably simple closed form. The practical middle ground is:

* fixed-node high-order quadrature on a transformed Gaussian domain;
* differentiate the quadrature rule itself;
* avoid adaptive node refinement inside the optimizer if you want a smooth outer objective.

A sufficient condition for a convergent Hermite-series treatment is roughly that (g^{-1}) lie in a Gaussian-weighted analytic class, e.g. admit an (L^2) Hermite expansion under the Gaussian measure. Entire functions with sub-Gaussian growth are the cleanest case. For non-entire links like SAS, fixed high-order quadrature is the safer implementation path.

---

# 20. Charbonnier / MM penalties (Q16, D8)

Let (t=A\beta), and for the MM surrogate
[
W_{MM}(t)=\operatorname{diag}\left(\frac{\varepsilon}{2\sqrt{t_m^2+\varepsilon^2}}\right).
]

Then
[
S_{\rm eff}(\beta)=A^\top W_{MM}(t)A.
]

### First directional derivative

Yes, your formula is correct:
[
\boxed{
D_\beta S_{\rm eff}[u]
======================

A^\top \operatorname{diag}\big(w'(t)\odot (Au)\big)A,
}
]
with
[
w'(t_m)=-\frac{\varepsilon t_m}{2(t_m^2+\varepsilon^2)^{3/2}}.
]

### Second directional derivative

[
\boxed{
D_\beta^2 S_{\rm eff}[u,v]
==========================

A^\top \operatorname{diag}\big(w''(t)\odot (Au)\odot(Av)\big)A,
}
]
with
[
w''(t_m)=\frac{\varepsilon(2t_m^2-\varepsilon^2)}{2(t_m^2+\varepsilon^2)^{5/2}}.
]

### True Charbonnier Hessian

For
[
\psi(t)=\varepsilon(\sqrt{t^2+\varepsilon^2}-\varepsilon),
]
[
\psi''(t)=\frac{\varepsilon^3}{(t^2+\varepsilon^2)^{3/2}}.
]

So
[
D_\beta\big[D_\beta^2 P\big][u]
===============================

A^\top \operatorname{diag}\big(\psi'''(t)\odot(Au)\big)A,
]
with
[
\psi'''(t)= -\frac{3\varepsilon^3 t}{(t^2+\varepsilon^2)^{5/2}}.
]

If you also want the second directional derivative of the true Hessian,
[
D_\beta^2\big[D_\beta^2P\big][u,v]
==================================

A^\top \operatorname{diag}\big(\psi''''(t)\odot(Au)\odot(Av)\big)A,
]
where
[
\psi''''(t)=\frac{3\varepsilon^3(4t^2-\varepsilon^2)}{(t^2+\varepsilon^2)^{7/2}}.
]

So the derivative provider needs **third** and **fourth** derivatives of (\psi), not fifth.

At exact MM convergence to the true minimizer, the envelope theorem is fine. At finite inner tolerance (|r|\le \delta), the profiled-gradient bias is
[
r^\top \beta_i,
]
hence
[
|r^\top\beta_i|
\le
|r|,|H^{-1}g_i|
\le \delta,|H^{-1}g_i|.
]
So the bias is (O(\delta)).

---

# 21. Stable reparameterization and sparse traces (Q17)

A blockwise version of stable reparameterization is the right compromise when each smooth has its own penalty block. That preserves global sparsity much better than a full dense rotation.

A hybrid approach is mathematically fine:

* solve the inner PIRLS/Newton system in the stable parameterization;
* map (\hat\beta) back;
* compute LAML derivatives in the original sparse parameterization.

This is consistent **provided the transform is (\theta)-independent**. If the transform depends on (\theta), then its derivatives enter the chain rule.

Trace contractions are invariant under a fixed congruence transform, so selected-inversion in the original sparse basis is fine even if the inner solve used a rotated basis. Wood’s 2011 stable method is explicitly about numerical robustness under such reparameterizations. ([University of Bath Personal Homepages][2])

---

# 22. Tierney–Kadane correction under design motion (Q18)

For large (n), I would not differentiate the TK correction with respect to (\tau) analytically unless you truly need it. The TK term is already (O(n^{-1})) relative to the leading Laplace term, so its (\tau)-gradient is also lower order.

If (\dim(\tau)) is small, central finite differences in (\tau) are entirely reasonable.

Your leverage derivative formula is almost right. For
[
s_m = x_m^\top H^{-1} w_m x_m,
]
if both (x_m) and (w_m) depend on (\tau), then
[
\boxed{
\dot s_m
========

2\dot x_m^\top H^{-1} w_m x_m
+
x_m^\top H^{-1}\dot w_m x_m
---------------------------

x_m^\top H^{-1}\dot H,H^{-1} w_m x_m.
}
]
If (w_m) is constant, this reduces to the expression you wrote.

---

# 23. Novel methods (N1–N4)

## 23.1 Stochastic trace estimation (N1)

For any matrix (A), Hutchinson with probes (z_m) satisfying (E[z_m z_m^\top]=I),
[
\hat t = \frac1M\sum_{m=1}^M z_m^\top A z_m
]
is unbiased for (\operatorname{tr}(A)).

For Gaussian probes and symmetric (A),
[
\operatorname{Var}(\hat t)=\frac{2}{M}|A|_F^2.
]
For general (A),
[
\operatorname{Var}(\hat t)=\frac{2}{M}|\operatorname{sym}(A)|_F^2.
]

So for the gradient trace (A=H^{-1}\dot H_k),
[
\operatorname{sd}(\hat t_k)
===========================

\sqrt{\frac{2}{M}},
|\operatorname{sym}(H^{-1}\dot H_k)|_F.
]

A useful rule is
[
M \gtrsim
\frac{2|\operatorname{sym}(H^{-1}\dot H_k)|_F^2}
{\varepsilon^2,\operatorname{tr}(H^{-1}\dot H_k)^2}
]
to get relative RMS error about (\varepsilon).

Yes, the **same probes** can be reused across all (K) gradient traces, so the gradient cost is (O(M)) solves, not (O(MK)).

For the Hessian double-trace term,
[
t_{ij}=\operatorname{tr}(H^{-1}\dot H_j H^{-1}\dot H_i),
]
the estimator
[
\hat t_{ij}=\frac1M\sum_m z_m^\top H^{-1}\dot H_j H^{-1}\dot H_i z_m
]
is also unbiased, with Gaussian variance
[
\operatorname{Var}(\hat t_{ij})
===============================

\frac{2}{M}\Big|
\operatorname{sym}\big(H^{-1}\dot H_j H^{-1}\dot H_i\big)
\Big|_F^2.
]

Probe reuse still works, but unlike the gradient, the exact computation of all (t_{ij}) generally does **not** collapse to (O(M)) solves: you still need the actions of (H^{-1}\dot H_i) on the probes, so the naive exact cost is (O(MK)) solves unless you add another approximation.

## 23.2 Natural gradient (N2)

Near a good Laplace approximation, (\nabla_\theta^2 V) is a reasonable approximation to the marginal Fisher information in (\theta). It breaks down when:

* Laplace is poor,
* constraints/rank changes create nonsmoothness,
* (H) is indefinite,
* or the outer posterior is strongly skewed.

In Gaussian REML there is a closed-form information expression in mixed-model covariance notation:
[
\mathcal I_{ij}
===============

\frac12 \operatorname{tr}(P V_i P V_j),
]
with (V_i=\partial V_y/\partial \theta_i) and (P) the REML projector. So yes, Gaussian REML admits a natural metric without forming the full profiled Hessian.

A trust-region in that metric is usually better behaved near (\lambda_k\to 0) or (\infty) than Euclidean steps.

## 23.3 Pólya–Gamma exact marginalization (N3)

Pólya–Gamma augmentation makes (\beta\mid \omega,y) Gaussian, but the remaining integral over (\omega\in\mathbb R_+^n) is still (n)-dimensional. I do not know a practical deterministic exact (O(p^3)) algorithm for the full Bernoulli-logit marginal likelihood from this route. The augmentation moves the intractability from (\beta) to (\omega), rather than removing it.

Formally the gradient is
[
\partial_\rho \log p(y\mid \rho)
================================

E_{\omega\mid y,\rho}\big[\partial_\rho \log p(y,\omega\mid \rho)\big],
]
but computing that expectation is again an inference problem.

So PG is excellent for conditional sampling, less attractive as an exact deterministic marginal-likelihood engine.

## 23.4 Model comparison by LAML (N4)

LAML is a Laplace approximation to the log marginal likelihood, so Bayes-factor style comparisons are sensible when:

* both models use the **same prior normalization convention**;
* the Laplace approximation is accurate;
* there are no boundary/rank-change pathologies.

Nested-model drops via (\lambda_k\to\infty) are boundary problems, so the LAML difference is not literally a standard LRT, though it is closely related to the same complexity/EDF tradeoff.

The key comparability issue is penalty normalization. The principled fix is: compare models only after putting them on the same prior scale, i.e. treat each penalty as the precision of the same underlying Gaussian prior on the penalized subspace. A practical normalization is to determinant-normalize each penalty on its positive subspace so that arbitrary basis scaling is removed. Without that, raw LAML values across models are not comparable.

---

The highest-value implementation change is:

[
\boxed{
\text{unify on }(a_i,g_i,B_i,\ell^S_i)\text{ and }(a_{ij},g_{ij},B_{ij},\ell^S_{ij}),
\text{ then add }M_i[u]=D_\beta B_i[u].
}
]

That gives you one evaluator for (\rho), (\psi), and later any other smooth outer hyperparameter.

[1]: https://webhomes.maths.ed.ac.uk/~swood34/gsm.pdf "https://webhomes.maths.ed.ac.uk/~swood34/gsm.pdf"
[2]: https://people.bath.ac.uk/man54/SAMBa/ITTs/ITT2/EDF/MultipleSmoothingParameterEstimationWood2012.pdf "Stable and Efficient Multiple Smoothing Parameter Estimation for Generalized Additive Models"
[3]: https://academic.oup.com/biomet/article-pdf/80/1/27/616333/80-1-27.pdf "https://academic.oup.com/biomet/article-pdf/80/1/27/616333/80-1-27.pdf"
[4]: https://web.mit.edu/haihao/www/papers/ALO.pdf "https://web.mit.edu/haihao/www/papers/ALO.pdf"
[5]: https://arxiv.org/abs/1803.05878 "https://arxiv.org/abs/1803.05878"

