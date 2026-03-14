I’ll fix the two notation issues first, because they affect several formulas.

For a smooth scalar objective (F(\beta)), let
[
H(\beta)=\nabla_\beta^2 F(\beta),\qquad C[u]=D_\beta H[u].
]
Then (C[u]v) and (C[v]u) are both (p)-vectors with components
[
[C[u]v]*a=\sum*{b,c}\frac{\partial^3 F}{\partial \beta_a\partial\beta_b\partial\beta_c},u_b v_c.
]
Because the third derivative tensor is symmetric in (b,c),
[
\boxed{C[u]v=C[v]u}
]
as vectors. For the scalar-GLM form
[
C[u]v=X^\top!\big(c\odot (Xu)\odot (Xv)\big),
]
this is immediate since elementwise multiplication is commutative. So the algorithmic line
[
\frac12{C(\beta_i)\beta_j+C(\beta_j)\beta_i}
]
is correct but redundant; in exact arithmetic it equals either term alone. That resolves E1–E2.

The rest below is the missing implementable material. The overall organization still matches the “fixed-(\beta) objects + generic correction operators” framework used in Wood’s general smooth-model REML/LAML setup. ([School of Mathematics][1])

---

## 1. Profiled Gaussian Hessian for general (\theta=(\rho,\psi))

Let
[
\nu=n-M_p,\qquad \hat\phi(\theta)=\frac{D_p(\hat\beta(\theta),\theta)}{\nu},
]
and
[
V(\theta)=\frac{\nu}{2}\log \hat\phi(\theta)+\frac12\log|H|*+-\frac12\log|S|*+ + c.
]

Write
[
G(\theta):=\frac{\nu}{2}\log\hat\phi(\theta).
]

At the mode, since (D_p=2F_p+\text{const}), where (F_p) is the penalized deviance-half,
[
\partial_i D_p = 2 a_i,
]
so
[
\boxed{\hat\phi_i:=\frac{\partial\hat\phi}{\partial\theta_i}=\frac{2a_i}{\nu}.}
]

Also, the profiled second derivative of (D_p/2) is the same envelope-Hessian piece as usual:
[
\partial_{ij}\Big(\frac{D_p}{2}\Big)=a_{ij}-g_i^\top H^{-1}g_j.
]
Hence
[
\boxed{
G_i=\frac{a_i}{\hat\phi},
}
]
and
[
\boxed{
G_{ij}
======

\frac{a_{ij}-g_i^\top H^{-1}g_j}{\hat\phi}
-\frac{2a_i a_j}{\nu \hat\phi^2}.
}
]

Therefore the full profiled-Gaussian outer Hessian is
[
\boxed{
V_{ij}
======

\frac{a_{ij}-g_i^\top H^{-1}g_j}{\hat\phi}
-\frac{2a_i a_j}{\nu \hat\phi^2}
+\frac12\Big[
\operatorname{tr}(H^{-1}\ddot H_{ij})
-------------------------------------

\operatorname{tr}(H^{-1}\dot H_j H^{-1}\dot H_i)
\Big]
-\frac12 \ell^S_{ij}.
}
]

Under the convention used in your question, this is the complete answer to A4.

One caveat: if your implementation defines the Laplace Hessian as (H^\star=H/\hat\phi) rather than using the unscaled (H), then the (\frac12\log|H^\star|) term contributes an additional (-\frac{r_H}{2}\log\hat\phi), with (r_H=\operatorname{rank}(H^\star)). Under your stated convention, there is no extra rank term.

---

## 2. Log-barrier LAML

Let
[
F_\tau(\beta,\theta)=F(\beta,\theta)-\tau\sum_{j\in\mathcal C}\log(\beta_j-b_j).
]
Define the slack vector on constrained coordinates
[
\Delta_j=\beta_j-b_j,\qquad j\in\mathcal C.
]

### 2.1 Barrier derivatives

Barrier gradient:
[
\nabla_\beta B_\tau(\beta) = -\tau,d^{(1)},
\qquad
d^{(1)}_j=
\begin{cases}
\Delta_j^{-1},& j\in\mathcal C,\
0,& j\notin\mathcal C.
\end{cases}
]

Barrier Hessian:
[
\boxed{
H_{\rm bar}(\beta)=\tau,D^{(2)},
\qquad
D^{(2)}=\operatorname{diag}(d^{(2)}),
\quad
d^{(2)}_j=
\begin{cases}
\Delta_j^{-2},& j\in\mathcal C,\
0,& j\notin\mathcal C.
\end{cases}
}
]

Directional derivative of the barrier Hessian:
[
\boxed{
D_\beta H_{\rm bar}[u]
======================

-2\tau,\operatorname{diag}!\big(u\odot d^{(3)}\big),
\qquad
d^{(3)}_j=
\begin{cases}
\Delta_j^{-3},& j\in\mathcal C,\
0,& j\notin\mathcal C.
\end{cases}
}
]

Second directional derivative:
[
\boxed{
D_\beta^2 H_{\rm bar}[u,v]
==========================

6\tau,\operatorname{diag}!\big(u\odot v\odot d^{(4)}\big),
\qquad
d^{(4)}_j=
\begin{cases}
\Delta_j^{-4},& j\in\mathcal C,\
0,& j\notin\mathcal C.
\end{cases}
}
]

### 2.2 Barrier-augmented Hessian and LAML

At the barrier mode (\hat\beta_\tau),
[
\boxed{
H_\tau = H + H_{\rm bar}
= H + \tau,D^{(2)}.
}
]

The barrier-augmented LAML is
[
\boxed{
V_\tau(\theta)
==============

F_\tau(\hat\beta_\tau,\theta)
+\frac12\log|H_\tau|
-\frac12\log|S|_+ + c.
}
]

### 2.3 Profiled gradient

If the constraints themselves do not depend on (\theta), then the barrier contributes no explicit fixed-(\beta) hyper-derivative:
[
a_i^\tau = a_i,\qquad g_i^\tau=g_i,\qquad B_i^\tau=B_i.
]
Only the mode-response operator and Hessian drift change:
[
\boxed{
\beta_i^\tau = -H_\tau^{-1} g_i,
}
]
and
[
\boxed{
\dot H_i^\tau
=============

B_i + D_\beta H_{\rm orig}[\beta_i^\tau] + D_\beta H_{\rm bar}[\beta_i^\tau].
}
]

So for a pure (\rho_k) direction,
[
\boxed{
\frac{\partial V_\tau}{\partial \rho_k}
=======================================

a_k
+\frac12\operatorname{tr}!\left(
H_\tau^{-1}\Big[
A_k + C_{\rm orig}[\beta_k^\tau]
-2\tau,\operatorname{diag}\big(\beta_k^\tau\odot d^{(3)}\big)
\Big]\right)
-\frac12 \ell^S_k.
}
]

So the barrier does **not** add an explicit new (g_k)-like term. It only modifies (H), the mode response, and the Hessian-drift correction.

### 2.4 What to do with (\tau)

Do **not** profile (\tau) jointly with (\theta) if the barrier is being used as an interior-point surrogate for a hard inequality. Then (\tau) is an algorithmic continuation parameter, not a statistical hyperparameter.

Use one of these two interpretations:

* If the barrier is only a numerical device for the hard-constrained problem, **anneal (\tau\downarrow 0)**.
* If the barrier is the actual model/penalty you want, hold (\tau) fixed and include it as part of the objective definition.

---

## 3. MM fixed-point differentiation and which Hessian belongs in (\log|H|)

Let the MM map be
[
T(\beta;\theta)=M(\beta,\theta)^{-1} r(\theta),
]
with
[
M(\beta,\theta)=K(\theta)+A^\top W(t)A,\qquad t=A\beta,
]
and fixed-point equation
[
\hat\beta = T(\hat\beta;\theta).
]

I will use the tangent-majorizer convention
[
\frac12 t^\top W(t)t
\quad\text{with}\quad
W(t)=\operatorname{diag}(w(t)),
\quad
w(t)=\frac{\psi'(t)}{t}.
]
For the Charbonnier penalty this gives
[
\psi(t)=\varepsilon(\sqrt{t^2+\varepsilon^2}-\varepsilon),\qquad
\psi'(t)=\frac{\varepsilon t}{\sqrt{t^2+\varepsilon^2}},
]
so
[
\boxed{
w(t)=\frac{\varepsilon}{\sqrt{t^2+\varepsilon^2}}.
}
]
If you keep the extra factor (1/2) inside (W_{MM}), multiply the formulas below accordingly.

### 3.1 (\nabla_\beta T)

For a perturbation (u),
[
D_\beta T[u]
============

-M^{-1}\big(D_\beta M[u]\big),T.
]
Since
[
D_\beta M[u]
============

A^\top \operatorname{diag}\big(w'(t)\odot Au\big)A,
]
and at the fixed point (T=\hat\beta), (A\hat\beta=t),
[
\big(D_\beta M[u]\big)\hat\beta
===============================

A^\top \operatorname{diag}\big(t\odot w'(t)\big) Au.
]
Hence the Jacobian operator is
[
\boxed{
\nabla_\beta T
==============

* M^{-1} A^\top \operatorname{diag}\big(t\odot w'(t)\big) A.
  }
  ]

### 3.2 Relation to the true Hessian

Because (\psi'(t)=w(t)t),
[
\psi''(t)=w(t)+t,w'(t).
]
Therefore the true penalty Hessian is
[
H_P^{\rm true}
==============

# A^\top \operatorname{diag}\big(\psi''(t)\big)A

A^\top \operatorname{diag}\big(w(t)+t w'(t)\big)A.
]
So
[
A^\top \operatorname{diag}(t w'(t))A
====================================

H_P^{\rm true}-A^\top W(t)A.
]

Now let
[
H_{\rm true}=K + H_P^{\rm true},\qquad
H_{MM}=M=K+A^\top W(t)A.
]
Then
[
I-\nabla_\beta T
================

# I+M^{-1}\big(H_P^{\rm true}-A^\top W A\big)

M^{-1}H_{\rm true}.
]
Hence
[
\boxed{
(I-\nabla_\beta T)^{-1}
=======================

H_{\rm true}^{-1} H_{MM}.
}
]

So the answer to A2/Q16c is yes:
[
\boxed{
(I-\nabla_\beta T)^{-1}=H_{\rm true}^{-1}H_{MM}.
}
]

### 3.3 (\nabla_\theta T)

For any hyper-direction (h),
[
\boxed{
D_\theta T[h]
=============

M^{-1}\Big(D_\theta r[h]-D_\theta M[h]\hat\beta\Big).
}
]
Therefore
[
\boxed{
\frac{d\hat\beta}{d\theta}[h]
=============================

# (I-\nabla_\beta T)^{-1}D_\theta T[h]

H_{\rm true}^{-1}\Big(D_\theta r[h]-D_\theta M[h]\hat\beta\Big).
}
]

That is exactly the same response as differentiating the true stationarity equation.

### 3.4 Which Hessian belongs in (\frac12\log|H|)?

The Laplace approximation is to the **true** penalized target
[
\exp{-F_{\rm true}(\beta,\theta)}.
]
Therefore its curvature term is always
[
\boxed{
H_{\rm true}=\nabla_\beta^2 F_{\rm true}(\hat\beta,\theta),
}
]
not the surrogate (H_{MM}).

There is no simple multiplicative correction factor taking (\log|H_{MM}|) to (\log|H_{\rm true}|). The exact difference is simply
[
\frac12\log|H_{\rm true}|-\frac12\log|H_{MM}|.
]

For the Charbonnier penalty,
[
\psi''(t)=\frac{\varepsilon^3}{(t^2+\varepsilon^2)^{3/2}},
\qquad
\psi'''(t)= -\frac{3\varepsilon^3 t}{(t^2+\varepsilon^2)^{5/2}}.
]
So if (F=-\ell+\lambda\sum_m \psi(t_m)), then
[
\boxed{
H_{\rm true}
============

H_L
+
\lambda A^\top \operatorname{diag}!\left(\frac{\varepsilon^3}{(t^2+\varepsilon^2)^{3/2}}\right) A.
}
]
Its directional derivative is
[
\boxed{
D_\beta H_{\rm true}[u]
=======================

D_\beta H_L[u]
+
\lambda A^\top \operatorname{diag}!\left(
-\frac{3\varepsilon^3 t}{(t^2+\varepsilon^2)^{5/2}}\odot Au
\right) A.
}
]

So the correct (\rho)-gradient is
[
\boxed{
V_\rho
======

\lambda P(\hat\beta)
+\frac12\operatorname{tr}!\left(H_{\rm true}^{-1}\dot H_\rho^{\rm true}\right)
-\partial_\rho \log Z_\lambda,
}
]
with
[
g_\rho=\lambda \nabla_\beta P(\hat\beta),
\qquad
\beta_\rho=-H_{\rm true}^{-1}g_\rho,
]
[
\dot H_\rho^{\rm true}
======================

\lambda H_P^{\rm true}
+
D_\beta H_L[\beta_\rho]
+
\lambda A^\top \operatorname{diag}\big(\psi'''(t)\odot A\beta_\rho\big) A.
]

Important: for a nonquadratic penalty, the Gaussian-prior normalization term (-\frac12\log|S|*+) is not the correct prior normalizer. For the true Charbonnier prior, the exact prior normalization is (-\log Z*\lambda), not (-\frac12\log|S|*+). So using (H*{\rm true}) with the quadratic (-\frac12\log|S|_+) term is not the exact marginal likelihood of the Charbonnier model; it is a hybrid approximation.

---

## 4. Generic moving-design GLM formulas, including (M_j[u])

For a scalar-predictor GLM with moving design (X(\psi)), let
[
\eta = X\beta,\qquad
H_L = X^\top W X,
]
where
[
W=\operatorname{diag}(w(\eta)),\quad
c(\eta)=w'(\eta),\quad
d(\eta)=w''(\eta).
]

For outer direction (\psi_j), define
[
\dot X_j=\frac{\partial X}{\partial\psi_j},\qquad
\dot\eta_j=\dot X_j \beta.
]

For coefficient-direction (u), define
[
\delta\eta = Xu,\qquad
\delta\dot\eta_j=\dot X_j u.
]

### 4.1 Fixed-(\beta) Hessian drift

[
\boxed{
B_j
===

\dot X_j^\top W X
+
X^\top W \dot X_j
+
X^\top \operatorname{diag}(c\odot \dot\eta_j),X.
}
]

There are exactly three terms. There are no extra (\dot X_j^\top(\cdots)\dot X_j) terms at first order in (\psi_j).

### 4.2 The missing callback (M_j[u]=D_\beta B_j[u])

Differentiate each term with respect to (\beta):
[
\boxed{
M_j[u]
======

\dot X_j^\top \operatorname{diag}(c\odot \delta\eta),X
+
X^\top \operatorname{diag}(c\odot \delta\eta),\dot X_j
+
X^\top \operatorname{diag}!\big(d\odot \delta\eta\odot \dot\eta_j + c\odot \delta\dot\eta_j\big),X.
}
]

This is the complete generic formula for B1(a), B1(c), and D11.

### 4.3 Family plugs

Binomial-logit with (y\sim\mathrm{Bin}(n,p)), (p=\operatorname{logit}^{-1}(\eta)):
[
w=n,p(1-p),\qquad
c=n,p(1-p)(1-2p),\qquad
d=n,p(1-p)(1-6p+6p^2).
]
So
[
\boxed{
B_j
===

\dot X_j^\top D(w)X + X^\top D(w)\dot X_j + X^\top D(c\odot \dot\eta_j)X,
}
]
[
\boxed{
M_j[u]
======

\dot X_j^\top D(c\odot \delta\eta)X
+
X^\top D(c\odot \delta\eta)\dot X_j
+
X^\top D(d\odot \delta\eta\odot \dot\eta_j + c\odot \delta\dot\eta_j)X.
}
]

Poisson-log:
[
w=\mu=e^\eta,\qquad c=\mu,\qquad d=\mu.
]

Gamma-log, observed information:
[
w=\alpha y e^{-\eta},\qquad c=-\alpha y e^{-\eta}=-w,\qquad d=\alpha y e^{-\eta}=w.
]

So for Gamma-log observed:
[
M_j[u]
======

\dot X_j^\top D(-w,\delta\eta)X
+
X^\top D(-w,\delta\eta)\dot X_j
+
X^\top D\big(w,\delta\eta,\dot\eta_j - w,\delta\dot\eta_j\big)X.
]

---

## 5. General noncanonical exponential-dispersion (d_{\rm obs})

Let (h(\eta)=g^{-1}(\eta)), (\mu=h(\eta)), and
[
h_1=h'(\eta),\quad h_2=h''(\eta),\quad h_3=h'''(\eta),\quad h_4=h''''(\eta),
]
[
V=V(\mu),\quad V_1=V'(\mu),\quad V_2=V''(\mu),\quad V_3=V'''(\mu),
]
[
f=y-\mu.
]

For exponential-dispersion families,
[
\partial_\eta \ell = \frac{f h_1}{\phi V}.
]

Define
[
A=\frac{h_1^2}{V},\qquad
B=\frac{h_2 V - h_1^2 V_1}{V^2}.
]
Then
[
w_{\rm obs}=\frac{A-fB}{\phi}.
]

The Fisher derivatives are
[
c_F=\frac{2V h_1 h_2 - V_1 h_1^3}{\phi V^2},
]
[
d_F=\frac{
2V^2 h_1 h_3 + 2V^2 h_2^2 - 5V V_1 h_1^2 h_2 - V V_2 h_1^4 + 2V_1^2 h_1^4
}{\phi V^3}.
]

For the observed derivatives, define
[
B_\eta
======

\frac{
V^2 h_3 - 3V V_1 h_1 h_2 - V V_2 h_1^3 + 2V_1^2 h_1^3
}{V^3},
]
[
B_{\eta\eta}
============

\frac{
V^3 h_4
-4V^2 V_1 h_1 h_3
-3V^2 V_1 h_2^2
-6V^2 V_2 h_1^2 h_2
-V^2 V_3 h_1^4
+12V V_1^2 h_1^2 h_2
+6V V_1 V_2 h_1^4
-6V_1^3 h_1^4
}{V^4}.
]

Then
[
c_{\rm obs}
===========

\frac{1}{\phi}\Big(c_F\phi + h_1 B - f B_\eta\Big),
]
i.e.
[
\boxed{
c_{\rm obs}
===========

\frac{
2V h_1 h_2 - V_1 h_1^3 + h_1(h_2 V - h_1^2V_1) - f V B_\eta
}{\phi V^2},
}
]
and the second observed derivative is
[
\boxed{
d_{\rm obs}
===========

\frac{1}{\phi}\Big(d_F\phi + h_2 B + 2h_1 B_\eta - f B_{\eta\eta}\Big).
}
]

Expanded out,
[
\boxed{
d_{\rm obs}
===========

\frac{

* V^3 f h_4
  +4V^3 h_1 h_3
  +3V^3 h_2^2
  +4V^2V_1 f h_1 h_3
  +3V^2V_1 f h_2^2
  -12V^2V_1 h_1^2 h_2
  +6V^2V_2 f h_1^2 h_2
  -3V^2V_2 h_1^4
  +V^2V_3 f h_1^4
  -12V V_1^2 f h_1^2 h_2
  +6V V_1^2 h_1^4
  -6V V_1V_2 f h_1^4
  +6V_1^3 f h_1^4
  }{\phi V^4}.
  }
  ]

That is B3.

---

## 6. Missing families in D2/D3/D4

### 6.1 Binomial-logit, (y\sim\mathrm{Bin}(n,p))

Yes: it is Bernoulli-logit with an extra factor (n), or equivalently with response (y\in{0,\dots,n}).

[
\ell=\omega\left[y\eta - n\log(1+e^\eta)+\log\binom{n}{y}\right],
]
[
s=\omega(y-np),\qquad p=\frac{1}{1+e^{-\eta}}.
]
[
\boxed{
w=\omega n p(1-p),\quad
c=\omega n p(1-p)(1-2p),\quad
d=\omega n p(1-p)(1-6p+6p^2).
}
]
Observed = Fisher, canonical link.

### 6.2 Gaussian-log link, (y\sim N(\mu,\phi)), (\mu=e^\eta)

[
\ell = -\omega\frac{(y-\mu)^2}{2\phi} + c.
]
[
s=\omega\frac{(y-\mu)\mu}{\phi}.
]

Observed:
[
\boxed{
w_{\rm obs}=\omega\frac{\mu(2\mu-y)}{\phi},\quad
c_{\rm obs}=\omega\frac{\mu(4\mu-y)}{\phi},\quad
d_{\rm obs}=\omega\frac{\mu(8\mu-y)}{\phi}.
}
]

Fisher:
[
\boxed{
w_F=\omega\frac{\mu^2}{\phi},\quad
c_F=\omega\frac{2\mu^2}{\phi},\quad
d_F=\omega\frac{4\mu^2}{\phi}.
}
]

Noncanonical.

### 6.3 Gaussian-inverse link, (\mu=1/\eta)

[
\ell = -\omega\frac{(y-\eta^{-1})^2}{2\phi}+c.
]
[
s=\omega\frac{1-\eta y}{\phi \eta^3}.
]

Observed:
[
\boxed{
w_{\rm obs}=\omega\frac{3-2\eta y}{\phi \eta^4},\quad
c_{\rm obs}=6\omega\frac{\eta y-2}{\phi \eta^5},\quad
d_{\rm obs}=12\omega\frac{5-2\eta y}{\phi \eta^6}.
}
]

Fisher:
[
\boxed{
w_F=\omega\frac{1}{\phi\eta^4},\quad
c_F=-4\omega\frac{1}{\phi\eta^5},\quad
d_F=20\omega\frac{1}{\phi\eta^6}.
}
]

Noncanonical.

### 6.4 Ordinal / multinomial generalization

For a multi-predictor observation (i), let the per-observation predictor vector be (\zeta_i\in\mathbb R^{m_i}), Jacobian (J_i=\partial \zeta_i/\partial\beta), and per-observation negative-loglikelihood Hessian
[
W_i = -\nabla^2_{\zeta_i}\ell_i(\zeta_i)\in\mathbb R^{m_i\times m_i}.
]
Then
[
\boxed{
H_L = \sum_i J_i^\top W_i J_i.
}
]

If (J_i) is (\beta)-independent, then for direction (u),
[
\boxed{
D_\beta H_L[u]
==============

\sum_i J_i^\top C_i[u] J_i,
\qquad
C_i[u] = \sum_{c=1}^{m_i}\frac{\partial W_i}{\partial \zeta_{ic}},\delta\zeta_{ic}^{(u)}.
}
]
Similarly
[
\boxed{
D_\beta^2 H_L[u,v]
==================

\sum_i J_i^\top D_i[u,v] J_i,
\qquad
D_i[u,v]
========

\sum_{c,d}
\frac{\partial^2 W_i}{\partial \zeta_{ic}\partial \zeta_{id}}
,\delta\zeta_{ic}^{(u)}\delta\zeta_{id}^{(v)}.
}
]

So the scalar-weight formula (X^\top!\operatorname{diag}(c\odot Xu)X) is just the (m_i=1) special case.

---

## 7. Gaussian location-scale: (D_\beta H_L[u]), (D_\beta^2 H_L[u,v]), and moving-design (M_j[u])

Take the direct-(\sigma) parameterization from your question:
[
\mu = X_\mu\beta_\mu,\qquad
\sigma = X_\sigma\beta_\sigma,\qquad
r=\frac{y-\mu}{\sigma}.
]

The observed Hessian blocks are
[
H_{\mu\mu}=X_\mu^\top D(w_{\mu\mu})X_\mu,\quad w_{\mu\mu}=\sigma^{-2},
]
[
H_{\mu\sigma}=X_\mu^\top D(w_{\mu\sigma})X_\sigma,\quad w_{\mu\sigma}=2r,\sigma^{-2},
]
[
H_{\sigma\sigma}=X_\sigma^\top D(w_{\sigma\sigma})X_\sigma,\quad w_{\sigma\sigma}=(3r^2-1)\sigma^{-2}.
]

For coefficient-direction (u=(u_\mu,u_\sigma)), define
[
\delta\mu^{(u)}=X_\mu u_\mu,\qquad \delta\sigma^{(u)}=X_\sigma u_\sigma.
]

### 7.1 First directional derivative

[
[D_\beta H_L[u]]_{\mu\mu}
=========================

X_\mu^\top D(q_{\mu\mu}(u)) X_\mu,
\quad
q_{\mu\mu}(u)= -2,\frac{\delta\sigma^{(u)}}{\sigma^3}.
]

[
[D_\beta H_L[u]]_{\mu\sigma}
============================

X_\mu^\top D(q_{\mu\sigma}(u)) X_\sigma,
\quad
q_{\mu\sigma}(u)= -\frac{2\delta\mu^{(u)}+6r,\delta\sigma^{(u)}}{\sigma^3}.
]

[
[D_\beta H_L[u]]_{\sigma\sigma}
===============================

X_\sigma^\top D(q_{\sigma\sigma}(u)) X_\sigma,
\quad
q_{\sigma\sigma}(u)= \frac{-6r,\delta\mu^{(u)}+(2-12r^2)\delta\sigma^{(u)}}{\sigma^3}.
]

This is the complete B2 first-order correction.

### 7.2 Second directional derivative

For two directions (u,v), define (\delta\mu^{(u)},\delta\sigma^{(u)}) and (\delta\mu^{(v)},\delta\sigma^{(v)}).

Then
[
[D_\beta^2 H_L[u,v]]_{\mu\mu}
=============================

X_\mu^\top D(d_{\mu\mu}(u,v)) X_\mu,
]
with
[
\boxed{
d_{\mu\mu}(u,v)= 6,\frac{\delta\sigma^{(u)}\delta\sigma^{(v)}}{\sigma^4}.
}
]

[
[D_\beta^2 H_L[u,v]]_{\mu\sigma}
================================

X_\mu^\top D(d_{\mu\sigma}(u,v)) X_\sigma,
]
with
[
\boxed{
d_{\mu\sigma}(u,v)
==================

6,\frac{\delta\mu^{(u)}\delta\sigma^{(v)}+\delta\sigma^{(u)}\delta\mu^{(v)}}{\sigma^4}
+
24,\frac{r,\delta\sigma^{(u)}\delta\sigma^{(v)}}{\sigma^4}.
}
]

[
[D_\beta^2 H_L[u,v]]_{\sigma\sigma}
===================================

X_\sigma^\top D(d_{\sigma\sigma}(u,v)) X_\sigma,
]
with
[
\boxed{
d_{\sigma\sigma}(u,v)
=====================

6,\frac{\delta\mu^{(u)}\delta\mu^{(v)}}{\sigma^4}
+
24,\frac{r,[\delta\mu^{(u)}\delta\sigma^{(v)}+\delta\sigma^{(u)}\delta\mu^{(v)}]}{\sigma^4}
+
6,\frac{(-1+10r^2)\delta\sigma^{(u)}\delta\sigma^{(v)}}{\sigma^4}.
}
]

That is D10.

### 7.3 Moving-design (M_j[u])

Let (X_\mu(\psi),X_\sigma(\psi)) move with (\psi_j). Define
[
\dot X_{\mu j}=\partial_{\psi_j}X_\mu,\qquad
\dot X_{\sigma j}=\partial_{\psi_j}X_\sigma,
]
[
\dot\mu_j=\dot X_{\mu j}\beta_\mu,\qquad
\dot\sigma_j=\dot X_{\sigma j}\beta_\sigma,
]
[
\delta\dot\mu_j=\dot X_{\mu j}u_\mu,\qquad
\delta\dot\sigma_j=\dot X_{\sigma j}u_\sigma.
]

Then
[
B_{\mu\mu,j}
============

\dot X_{\mu j}^\top D(w_{\mu\mu})X_\mu
+
X_\mu^\top D(w_{\mu\mu})\dot X_{\mu j}
+
X_\mu^\top D(-2\dot\sigma_j/\sigma^3)X_\mu.
]

[
B_{\mu\sigma,j}
===============

\dot X_{\mu j}^\top D(w_{\mu\sigma})X_\sigma
+
X_\mu^\top D(w_{\mu\sigma})\dot X_{\sigma j}
+
X_\mu^\top D!\left(
-2\frac{\dot\mu_j}{\sigma^3}
-6r\frac{\dot\sigma_j}{\sigma^3}
\right)X_\sigma.
]

[
B_{\sigma\sigma,j}
==================

\dot X_{\sigma j}^\top D(w_{\sigma\sigma})X_\sigma
+
X_\sigma^\top D(w_{\sigma\sigma})\dot X_{\sigma j}
+
X_\sigma^\top D!\left(
-6r\frac{\dot\mu_j}{\sigma^3}
+
(2-12r^2)\frac{\dot\sigma_j}{\sigma^3}
\right)X_\sigma.
]

Now (M_j[u]=D_\beta B_j[u]). The blocks are:

[
M_{\mu\mu,j}[u]
===============

\dot X_{\mu j}^\top D(q_{\mu\mu}(u))X_\mu
+
X_\mu^\top D(q_{\mu\mu}(u))\dot X_{\mu j}
+
X_\mu^\top D(m_{\mu\mu,j}(u))X_\mu,
]
with
[
\boxed{
m_{\mu\mu,j}(u)
===============

## 6,\frac{\delta\sigma^{(u)}\dot\sigma_j}{\sigma^4}

2,\frac{\delta\dot\sigma_j}{\sigma^3}.
}
]

[
M_{\mu\sigma,j}[u]
==================

\dot X_{\mu j}^\top D(q_{\mu\sigma}(u))X_\sigma
+
X_\mu^\top D(q_{\mu\sigma}(u))\dot X_{\sigma j}
+
X_\mu^\top D(m_{\mu\sigma,j}(u))X_\sigma,
]
with
[
\boxed{
m_{\mu\sigma,j}(u)
==================

## 6,\frac{\delta\sigma^{(u)}\dot\mu_j+\delta\mu^{(u)}\dot\sigma_j+4r,\delta\sigma^{(u)}\dot\sigma_j}{\sigma^4}

## 2,\frac{\delta\dot\mu_j}{\sigma^3}

6r,\frac{\delta\dot\sigma_j}{\sigma^3}.
}
]

[
M_{\sigma\sigma,j}[u]
=====================

\dot X_{\sigma j}^\top D(q_{\sigma\sigma}(u))X_\sigma
+
X_\sigma^\top D(q_{\sigma\sigma}(u))\dot X_{\sigma j}
+
X_\sigma^\top D(m_{\sigma\sigma,j}(u))X_\sigma,
]
with
[
\boxed{
m_{\sigma\sigma,j}(u)
=====================

## \left(6\frac{\delta\mu^{(u)}}{\sigma^4}+24\frac{r,\delta\sigma^{(u)}}{\sigma^4}\right)\dot\mu_j

6r,\frac{\delta\dot\mu_j}{\sigma^3}
+
\left(24\frac{r,\delta\mu^{(u)}}{\sigma^4}+6\frac{(-1+10r^2)\delta\sigma^{(u)}}{\sigma^4}\right)\dot\sigma_j
+
(2-12r^2)\frac{\delta\dot\sigma_j}{\sigma^3}.
}
]

That completes B1(b).

For Firth in Gaussian location-scale, the Fisher information is
[
I_{\mu\mu}=X_\mu^\top D(\sigma^{-2})X_\mu,\qquad
I_{\mu\sigma}=0,\qquad
I_{\sigma\sigma}=X_\sigma^\top D(2\sigma^{-2})X_\sigma.
]
So (D_\alpha I[u]) is exactly the same structural correction with Fisher weights in place of observed weights. Hence the Firth-corrected outer gradient is indeed
[
\boxed{
V_k^{\rm Firth}
===============

V_k + \frac12 \operatorname{tr}(I^{-1}\dot I_k),
}
]
and yes, if your LAML uses observed information while Firth uses Fisher information, you need **both** correction operators (C[u]) and (C_F[u]). Firth’s bias-reduction/Jeffreys-prior equivalence in canonical exponential-family settings is the classical Firth result. ([OUP Academic][2])

---

## 8. Survival location-scale derivative provider (D7 / D9)

Use the notation

* (X_1): exit log-cumulative-hazard design, (\eta_1=X_1\beta_h)
* (X_0): entry log-cumulative-hazard design, (\eta_0=X_0\beta_h)
* (D): derivative design, (s=D\beta_h)
* (Z): log-scale design, (\eta_\sigma = Z\beta_\sigma)

and
[
E_1 = e^{\eta_1-\eta_\sigma},\qquad
E_0 = e^{\eta_0-\eta_\sigma}.
]

Per observation,
[
n_i := -\ell_i
==============

-\delta_i(\eta_{1i}+\log s_i-\eta_{\sigma i})
+E_{1i}-E_{0i}.
]

### 8.1 Joint Hessian

The gradient blocks are
[
\nabla_{\beta_h} n
==================

-\delta\odot\Big(X_1 + D/s\Big) + E_1\odot X_1 - E_0\odot X_0,
]
[
\nabla_{\beta_\sigma} n
=======================

(\delta - E_1 + E_0)\odot Z.
]

The Hessian blocks are
[
\boxed{
H_{hh}
======

## X_1^\top D(E_1) X_1

X_0^\top D(E_0) X_0
+
D^\top D(\delta/s^2) D,
}
]
[
\boxed{
H_{h\sigma}
===========

* X_1^\top D(E_1) Z

- X_0^\top D(E_0) Z,
  \qquad
  H_{\sigma h}=H_{h\sigma}^\top,
  }
  ]
  [
  \boxed{
  H_{\sigma\sigma}
  =
  Z^\top D(E_1-E_0) Z.
  }
  ]

### 8.2 First correction (D_\alpha H_L[u])

For (u=(u_h,u_\sigma)), define
[
\delta\eta_1^{(u)}=X_1 u_h,\qquad
\delta\eta_0^{(u)}=X_0 u_h,\qquad
\delta\eta_\sigma^{(u)}=Z u_\sigma,\qquad
\delta s^{(u)}=D u_h,
]
and
[
\xi_1^{(u)}=\delta\eta_1^{(u)}-\delta\eta_\sigma^{(u)},\qquad
\xi_0^{(u)}=\delta\eta_0^{(u)}-\delta\eta_\sigma^{(u)}.
]

Then
[
\boxed{
[D_\alpha H_L[u]]_{hh}
======================

## X_1^\top D(E_1\xi_1^{(u)})X_1

## X_0^\top D(E_0\xi_0^{(u)})X_0

2 D^\top D!\left(\delta,\delta s^{(u)}/s^3\right)D,
}
]

[
\boxed{
[D_\alpha H_L[u]]_{h\sigma}
===========================

* X_1^\top D(E_1\xi_1^{(u)}) Z

- X_0^\top D(E_0\xi_0^{(u)}) Z,
  }
  ]

[
\boxed{
[D_\alpha H_L[u]]_{\sigma\sigma}
================================

Z^\top D!\left(E_1\xi_1^{(u)} - E_0\xi_0^{(u)}\right) Z.
}
]

### 8.3 Second correction (D_\alpha^2 H_L[u,v])

For another direction (v), define (\xi_1^{(v)},\xi_0^{(v)},\delta s^{(v)}) analogously.

Then
[
\boxed{
[D_\alpha^2 H_L[u,v]]_{hh}
==========================

## X_1^\top D(E_1\xi_1^{(u)}\xi_1^{(v)})X_1

X_0^\top D(E_0\xi_0^{(u)}\xi_0^{(v)})X_0
+
6 D^\top D!\left(\delta,\delta s^{(u)}\delta s^{(v)}/s^4\right)D,
}
]

[
\boxed{
[D_\alpha^2 H_L[u,v]]_{h\sigma}
===============================

* X_1^\top D(E_1\xi_1^{(u)}\xi_1^{(v)}) Z

- X_0^\top D(E_0\xi_0^{(u)}\xi_0^{(v)}) Z,
  }
  ]

[
\boxed{
[D_\alpha^2 H_L[u,v]]_{\sigma\sigma}
====================================

Z^\top D!\left(E_1\xi_1^{(u)}\xi_1^{(v)} - E_0\xi_0^{(u)}\xi_0^{(v)}\right) Z.
}
]

That is the complete survival location-scale provider requested in A1 / D9.

### 8.4 Weight-channel view

This model is not a single (X_h^\top \operatorname{diag}(w_{hh})X_h) block. It has three (h)-channels: exit (X_1), entry (X_0), and derivative (D). The minimal channel set is
[
\mathcal C={1,0,\sigma,s}.
]
The nonzero scalar channel weights are:

* (w^{11}_{hh}=E_1)
* (w^{00}_{hh}=-E_0)
* (w^{ss}_{hh}=\delta/s^2)
* (w^{1\sigma}_{h\sigma}=-E_1)
* (w^{0\sigma}_{h\sigma}=+E_0)
* (w^{\sigma\sigma}_{\sigma\sigma}=E_1-E_0)

with first derivatives
[
\partial_{\eta_1}E_1=E_1,\quad
\partial_{\eta_\sigma}E_1=-E_1,\quad
\partial_{\eta_0}E_0=E_0,\quad
\partial_{\eta_\sigma}E_0=-E_0,\quad
\partial_s(\delta/s^2)=-2\delta/s^3,
]
and second derivatives
[
\partial_{\eta_1\eta_1}^2E_1=E_1,\quad
\partial_{\eta_1\eta_\sigma}^2E_1=-E_1,\quad
\partial_{\eta_\sigma\eta_\sigma}^2E_1=E_1,
]
[
\partial_{\eta_0\eta_0}^2(-E_0)=-E_0,\quad
\partial_{\eta_0\eta_\sigma}^2(-E_0)=+E_0,\quad
\partial_{\eta_\sigma\eta_\sigma}^2(-E_0)=-E_0,
]
[
\partial_{ss}^2(\delta/s^2)=6\delta/s^4.
]

---

## 9. Penalty pseudo-logdet under (\psi): first, second, and cross derivatives

Let (S(\theta)) be symmetric PSD with constant rank (r). Let (S^+) be the Moore–Penrose pseudoinverse, and let
[
P_+ = SS^+, \qquad P_0 = I-P_+.
]

### 9.1 First derivative

As long as rank is constant,
[
\boxed{
\ell_i^S:=\partial_i \log|S|_+ = \operatorname{tr}(S^+ S_i),
}
]
where (S_i=\partial_i S).

This remains correct even if the positive eigenspace rotates.

### 9.2 Second derivative: fixed positive subspace vs moving positive subspace

If the positive eigenspace is fixed, then the usual reduced-PD formula is correct:
[
\boxed{
\ell_{ij}^S
===========

## \operatorname{tr}(S^+ S_{ij})

\operatorname{tr}(S^+ S_i S^+ S_j).
}
]

If the positive eigenspace moves, that formula is incomplete. The correct constant-rank formula is
[
\boxed{
\ell_{ij}^S
===========

## \operatorname{tr}(S^+ S_{ij})

\operatorname{tr}(S^+ S_i S^+ S_j)
+
\operatorname{tr}(S^{+2} S_i P_0 S_j)
+
\operatorname{tr}(S^{+2} S_j P_0 S_i).
}
]

The last two projector-mixing terms are exactly what the simple formula misses when (\psi) rotates the positive/null decomposition.

So C4 is:

* first derivative: yes, (\operatorname{tr}(S^+S_j));
* second derivative: your simple formula is correct **only if** (P_0 S_j P_+=0), i.e. no positive-null mixing.

### 9.3 Cross ((\rho_k,\psi_j))

Write
[
A_k=\partial_{\rho_k} S,\qquad
A_{k,j}=\partial_{\psi_j}\partial_{\rho_k}S.
]
Then
[
\boxed{
\ell_{kj}^S
===========

## \operatorname{tr}(S^+ A_{k,j})

\operatorname{tr}(S^+ A_k S^+ S_j)
+
\operatorname{tr}(S^{+2}A_k P_0 S_j)
+
\operatorname{tr}(S^{+2}S_j P_0 A_k).
}
]

If (A_k) preserves the nullspace, i.e.
[
P_0 A_k = A_k P_0 = 0,
]
then the extra projector terms vanish and your proposed formula is correct:
[
\boxed{
\ell_{kj}^S
===========

## \operatorname{tr}(S^+ A_{k,j})

\operatorname{tr}(S^+ A_k S^+ S_j).
}
]

### 9.4 What if (\partial_{\psi_j}S) has nullspace components?

If every (S_k(\psi)) has a fixed nullspace (N), then for every (n\in N),
[
S_k(\psi)n=0\quad\forall\psi
;\Rightarrow;
\partial_{\psi_j}S_k(\psi),n=0.
]
Because the matrices are symmetric, also (n^\top \partial_{\psi_j}S_k(\psi)=0). So **true nullspace leakage is impossible when the nullspace is genuinely fixed**.

Therefore:

* if your algebra says (S_j) has nullspace components while the nullspace is supposed to be fixed, that is either

  * a basis-mismatch artifact, or
  * numerical leakage from finite differencing / finite precision.

In that case, project it out before applying the fixed-nullspace formulas:
[
S_j \leftarrow P_+ S_j P_+.
]

### 9.5 Efficient computation

The efficient way is to work in a basis (Q_+) for the positive subspace:
[
\bar S = Q_+^\top S Q_+,\qquad
\bar S_i = Q_+^\top S_i Q_+,\qquad
\bar S_{ij}=Q_+^\top S_{ij} Q_+.
]
Factor (\bar S = LL^\top) once. Then
[
\ell_i^S = \operatorname{tr}(L^{-1}\bar S_i L^{-\top}),
]
[
\ell_{ij}^S
===========

## \operatorname{tr}(L^{-1}\bar S_{ij}L^{-\top})

\operatorname{tr}(L^{-1}\bar S_i \bar S^{-1} \bar S_j L^{-\top})
]
if the positive subspace is fixed.

If it is not fixed, either use the projector-corrected formula above or work spectrally from the positive eigenpairs.

---

## 10. Smooth spectral regularization (r_\varepsilon): exact gradient and Hessian formulas

Define
[
r_\varepsilon(\sigma)=\frac12\left(\sigma+\sqrt{\sigma^2+4\varepsilon^2}\right).
]
Let
[
\mathcal R_\varepsilon(H)=U,\operatorname{diag}(r_\varepsilon(\sigma_a)),U^\top
]
for the eigendecomposition (H=U\operatorname{diag}(\sigma_a)U^\top).

Let
[
\phi(\sigma)=\log r_\varepsilon(\sigma).
]
Then
[
r'*\varepsilon(\sigma)=\frac12\left(1+\frac{\sigma}{\sqrt{\sigma^2+4\varepsilon^2}}\right),
]
and crucially
[
\boxed{
\phi'(\sigma)=\frac{r'*\varepsilon(\sigma)}{r_\varepsilon(\sigma)}
==================================================================

\frac{1}{\sqrt{\sigma^2+4\varepsilon^2}}.
}
]

So C1’s observation is correct: the derivative of the regularized logdet is **not**
(\operatorname{tr}(\mathcal R_\varepsilon(H)^{-1}\dot H)). The correct gradient operator is
[
\boxed{
G_\varepsilon(H)
================

# U\operatorname{diag}!\left(\frac{1}{\sqrt{\sigma_a^2+4\varepsilon^2}}\right)U^\top

(H^2+4\varepsilon^2 I)^{-1/2}.
}
]

Hence
[
\boxed{
\partial_i \log|\mathcal R_\varepsilon(H)|
==========================================

\operatorname{tr}!\big(G_\varepsilon(H),\dot H_i\big).
}
]

### 10.1 Hessian of the regularized logdet

For a spectral trace function (f(H)=\operatorname{tr}\phi(H)),
[
D^2 f_H[E,F]
============

\sum_{a,b}\phi^{[1]}(\sigma_a,\sigma_b),E'*{ab}F'*{ba},
\qquad
E'=U^\top E U,
]
where
[
\phi^{[1]}(x,y)=
\begin{cases}
\phi''(x),& x=y,[1mm]
\dfrac{\phi'(x)-\phi'(y)}{x-y},& x\neq y.
\end{cases}
]

Here
[
\phi''(\sigma)= -\frac{\sigma}{(\sigma^2+4\varepsilon^2)^{3/2}}.
]

So
[
\boxed{
\partial_{ij}\log|\mathcal R_\varepsilon(H)|
============================================

\operatorname{tr}!\big(G_\varepsilon(H),\ddot H_{ij}\big)
+
\sum_{a,b}\Gamma_{ab},
(\dot H_i')*{ab}(\dot H_j')*{ba},
}
]
with
[
\Gamma_{aa}
===========

-\frac{\sigma_a}{(\sigma_a^2+4\varepsilon^2)^{3/2}},
]
[
\Gamma_{ab}
===========

\frac{
(\sigma_a^2+4\varepsilon^2)^{-1/2}
----------------------------------

(\sigma_b^2+4\varepsilon^2)^{-1/2}
}{\sigma_a-\sigma_b}
====================

-\frac{\sigma_a+\sigma_b}{
\sqrt{\sigma_a^2+4\varepsilon^2},
\sqrt{\sigma_b^2+4\varepsilon^2},
\big(\sqrt{\sigma_a^2+4\varepsilon^2}+\sqrt{\sigma_b^2+4\varepsilon^2}\big)
}.
]

So the standard Hessian term
[
-\operatorname{tr}(H^{-1}\dot H_j H^{-1}\dot H_i)
]
must be replaced by this spectral divided-difference kernel.

### 10.2 Derivative of (\mathcal R_\varepsilon(H)^{-1})

Let
[
g(\sigma)=\frac1{r_\varepsilon(\sigma)}
=======================================

# \frac{2}{\sigma+\sqrt{\sigma^2+4\varepsilon^2}}

\frac{\sqrt{\sigma^2+4\varepsilon^2}-\sigma}{2\varepsilon^2}.
]
Then
[
g'(\sigma)= -\frac{1}{r_\varepsilon(\sigma)\sqrt{\sigma^2+4\varepsilon^2}}.
]

The Fréchet derivative is
[
\boxed{
D\big[\mathcal R_\varepsilon(H)^{-1}\big][E]
============================================

U\Big(G^{(g)}\circ E'\Big)U^\top,
}
]
where (E'=U^\top E U), (\circ) is Hadamard product, and
[
G^{(g)}*{aa}=g'(\sigma_a),\qquad
G^{(g)}*{ab}=\frac{g(\sigma_a)-g(\sigma_b)}{\sigma_a-\sigma_b}\quad (a\neq b).
]

If you need the second derivative of (\mathcal R_\varepsilon(H)^{-1}), use the standard second divided-difference form for spectral matrix functions:
[
\boxed{
D^2 g_H[E,F]
============

U\left(
\sum_c g^{[2]}(\sigma_a,\sigma_c,\sigma_b),E'*{ac}F'*{cb}
\right)_{ab}U^\top,
}
]
with the scalar second divided difference
[
g^{[2]}(x,y,z)
==============

\frac{g^{[1]}(x,y)-g^{[1]}(y,z)}{x-z}.
]

---

## 11. Can the (\beta_{ij}) solves be avoided?

Yes for the (\operatorname{tr}(H^{-1}C[\beta_{ij}])) part in scalar GLMs.

Let
[
C[b]=X^\top D(c\odot Xb)X.
]
Then
[
\operatorname{tr}(H^{-1}C[b])
=============================

\operatorname{tr}\big(XH^{-1}X^\top D(c\odot Xb)\big).
]
Define
[
h=\operatorname{diag}(XH^{-1}X^\top),\qquad
t=c\odot h.
]
Then
[
\boxed{
\operatorname{tr}(H^{-1}C[b]) = t^\top Xb = b^\top X^\top t.
}
]

Now if
[
\beta_{ij}=-H^{-1}r_{ij},
]
then
[
\boxed{
\operatorname{tr}(H^{-1}C[\beta_{ij}])
======================================

-r_{ij}^\top z_c,
\qquad
z_c := H^{-1}X^\top t.
}
]

So for scalar GLMs, the (\beta_{ij}) solve is not needed inside that trace term: one precomputed adjoint solve (z_c) replaces all pairwise solves.

The same idea extends blockwise: whenever your provider can reduce
[
\operatorname{tr}(H^{-1}D_\beta H_L[b])
]
to a linear functional of (b), precompute the corresponding adjoint vector(s) once and contract with (r_{ij}).

But in general you **cannot** eliminate (\beta_{ij}) everywhere from the exact Hessian unless the provider exposes those trace-contraction adjoints. Without them, computing (XH^{-1}r_{ij}) is still another solve or an implicit full hat operator.

---

## 12. ALO details

For observation (i), let (J_i) be the Jacobian from coefficients to that observation’s predictor block, (W_i) the per-observation NLL Hessian block, and
[
A_i := J_i H^{-1} J_i^\top.
]
Then the ALO shift on predictor scale is most stably written as
[
\boxed{
\Delta\eta_i^{\rm ALO}
======================

A_i(I-W_iA_i)^{-1}s_i,
}
]
where (s_i=\nabla_{\eta_i}\ell_i^{\rm NLL}). This is algebraically equivalent to the earlier ((I-\mathcal H_{ii})^{-1}\mathcal H_{ii}W_i^{-1}s_i) formula but does **not** require (W_i^{-1}).

So when (W_i) is singular, use this (A_i(I-W_iA_i)^{-1}s_i) form.

### 12.1 Computing all leverages

You do **not** need (Bn) separate solves. After one factorization of (H), compute the diagonal (B\times B) blocks of
[
A = J H^{-1} J^\top
]
using either

* sparse inverse subset / selected inversion, or
* multifrontal extraction of the required inverse entries.

This is the exact multi-block analogue of computing single-predictor leverages from (\operatorname{diag}(XH^{-1}X^\top W)). The modern ALO literature uses exactly this quadratic/influence viewpoint for twice-differentiable losses and regularizers. ([Proceedings of Machine Learning Research][3])

### 12.2 Approximate leave-one-out standard errors

Use the influence covariance
[
\Delta\eta_i \approx A_i(I-W_iA_i)^{-1}s_i.
]
Hence
[
\boxed{
\operatorname{Var}(\Delta\eta_i)
\approx
A_i(I-W_iA_i)^{-1}\operatorname{Var}(s_i)(I-A_iW_i)^{-1}A_i^\top.
}
]

Under the local quadratic model, (\operatorname{Var}(s_i)\approx W_i), giving
[
\boxed{
\operatorname{Var}(\Delta\eta_i)
\approx
A_i(I-W_iA_i)^{-1}W_i(I-A_iW_i)^{-1}A_i^\top.
}
]

The diagonal gives approximate componentwise LOO standard errors. A Cook-type scalar can be taken as
[
\boxed{
D_i^{\rm Cook\text{-}ALO}
=========================

\Delta\eta_i^\top W_i \Delta\eta_i.
}
]

---

## 13. Stochastic trace estimation details

Let
[
A_k = H^{-1}\dot H_k,
\qquad
S_k=\operatorname{sym}(A_k)=\frac12(A_k+A_k^\top).
]
Since (z^\top A_k z = z^\top S_k z), only the symmetric part matters.

### 13.1 Gaussian vs Rademacher

For Gaussian probes (z\sim N(0,I)),
[
\operatorname{Var}(z^\top A_k z)=2|S_k|_F^2.
]

For Rademacher probes (z_i\in{-1,+1}),
[
z^\top S_k z = \operatorname{tr}(S_k) + 2\sum_{a<b}(S_k)_{ab} z_a z_b,
]
so
[
\boxed{
\operatorname{Var}(z^\top A_k z)
================================

# 2\sum_{a\neq b}(S_k)_{ab}^2

2\Big(|S_k|*F^2-\sum_a (S_k)*{aa}^2\Big).
}
]

Hence Rademacher always removes the diagonal variance contribution and is strictly better unless (S_k) has zero diagonal. Recommendation:

[
\boxed{\text{Use Rademacher probes by default.}}
]

### 13.2 Adaptive probe count

For each probe (m), compute
[
q_{m,k}=z_m^\top A_k z_m.
]
Use the running sample mean
[
\bar q_{M,k}=\frac1M\sum_{m=1}^M q_{m,k},
]
and unbiased sample variance
[
s_{M,k}^2 = \frac{1}{M-1}\sum_{m=1}^M (q_{m,k}-\bar q_{M,k})^2.
]
Then the estimated Monte Carlo standard error is
[
\boxed{
\widehat{\operatorname{se}}(\hat t_k)
=====================================

\frac{s_{M,k}}{\sqrt{M}},
\qquad
\hat t_k=\bar q_{M,k}.
}
]

Stop when
[
\boxed{
\max_k
\frac{\widehat{\operatorname{se}}(\hat t_k)}
{\max(|\hat t_k|,\tau_{\rm rel})}
\le \varepsilon,
}
]
with a small (\tau_{\rm rel}) to protect near-zero traces.

### 13.3 Bias / error from truncated PCG

Suppose each approximate solve (\tilde x(z)) satisfies residual bound
[
|z-H\tilde x(z)|*2 \le \delta*{\rm PCG}|z|*2.
]
Then solve error
[
e(z):=\tilde x(z)-H^{-1}z = H^{-1}(z-H\tilde x(z))
]
satisfies
[
\boxed{
|e(z)|*2 \le \frac{\delta*{\rm PCG}}{\lambda*{\min}(H)}|z|_2.
}
]

So the probe-wise quadratic-form error obeys
[
|\tilde q_k(z)-q_k(z)|
======================

|e(z)^\top \dot H_k z|
\le
\frac{\delta_{\rm PCG}}{\lambda_{\min}(H)}
|\dot H_k|_2 |z|_2^2.
]

Taking expectations gives the absolute Monte Carlo perturbation bound
[
\boxed{
\mathbb E|\tilde q_k-q_k|
\le
\frac{\delta_{\rm PCG}}{\lambda_{\min}(H)}
|\dot H_k|_2,\mathbb E|z|_2^2
=============================

\frac{\delta_{\rm PCG},p}{\lambda_{\min}(H)}|\dot H_k|_2
}
]
for both Gaussian and Rademacher probes.

If the approximate inverse is a linear operator (\tilde H^{-1}), then
[
|\tilde H^{-1}-H^{-1}|*2 \le \frac{\delta*{\rm PCG}}{\lambda_{\min}(H)},
]
hence
[
\boxed{
\big|\operatorname{tr}((\tilde H^{-1}-H^{-1})\dot H_k)\big|
\le
\frac{\delta_{\rm PCG}}{\lambda_{\min}(H)}\sqrt{p},|\dot H_k|_F.
}
]

---

## 14. EFS and (\psi)

The Fellner–Schall / EFS coordinate update relies on the special penalty-only structure of (\rho):

* (A_k=\partial_{\rho_k}S) is PSD,
* the score term is (\lambda_k\beta^\top S_k\beta),
* and the update acts multiplicatively on (\lambda_k).

For a generic (\psi_j), (B_{\psi_j}) contains design-motion and likelihood-curvature terms and need not be PSD or even sign-definite. There is no direct coordinatewise positive update analogous to
[
\rho_k^{\rm new}
================

\rho_k+\frac{\lambda_k\beta^\top S_k\beta-\operatorname{tr}(H^{-1}A_k)}
{\operatorname{tr}(H^{-1}A_kH^{-1}A_k)}.
]

So the answer to C7 is:

[
\boxed{
\text{No, EFS does not directly generalize to }\psi\text{ by }A_k\mapsto B_{\psi_j}.
}
]

The closest generic approximation is a Gauss–Newton outer step using only the trace-curvature piece
[
G_{ij}^{\rm GN}\approx \frac12\operatorname{tr}(H^{-1}B_j H^{-1}B_i),
]
but that is just a quasi-Newton approximation, not an EFS fixed-point update.

---

## 15. CLogLog Gaussian convolution

Let
[
g(\eta)=1-e^{-e^\eta},
\qquad
L(\mu,\sigma)=\mathbb E[g(\mu+\sigma Z)],\quad Z\sim N(0,1).
]
Then
[
1-L(\mu,\sigma)=\mathbb E[e^{-Y}],\qquad Y\sim \mathrm{Lognormal}(\mu,\sigma^2).
]

### 15.1 Mellin–Barnes representation

Using
[
e^{-y}
======

\frac{1}{2\pi i}
\int_{c-i\infty}^{c+i\infty}\Gamma(s),y^{-s},ds,
\qquad c>0,
]
and
[
\mathbb E(Y^{-s}) = e^{-s\mu + \frac12 s^2\sigma^2},
]
we get
[
\boxed{
\mathbb E[e^{-Y}]
=================

\frac{1}{2\pi i}
\int_{c-i\infty}^{c+i\infty}
\Gamma(s),
\exp!\left(-s\mu+\frac12 s^2\sigma^2\right),ds.
}
]

So
[
\boxed{
L(\mu,\sigma)
=============

1-
\frac{1}{2\pi i}
\int_{c-i\infty}^{c+i\infty}
\Gamma(s),
\exp!\left(-s\mu+\frac12 s^2\sigma^2\right),ds.
}
]

### 15.2 Saddle point

Define
[
\Phi(s)=\log\Gamma(s)-\mu s + \frac12 \sigma^2 s^2.
]
The saddle (s^\star) solves
[
\boxed{
\Phi'(s^\star)=\psi_0(s^\star)-\mu+\sigma^2 s^\star = 0,
}
]
where (\psi_0) is the digamma function.

A good starting approximation comes from replacing (\psi_0(s)) by (\log s):
[
\boxed{
s_0 \approx \frac{1}{\sigma^2}W(\sigma^2 e^\mu),
}
]
where (W) is Lambert (W). Then refine by Newton:
[
s \leftarrow s - \frac{\psi_0(s)-\mu+\sigma^2 s}{\psi_1(s)+\sigma^2}.
]

The leading saddle approximation is
[
\boxed{
\mathbb E[e^{-Y}]
\approx
\frac{\Gamma(s^\star)\exp!\left(-\mu s^\star+\frac12\sigma^2{s^\star}^2\right)}
{\sqrt{2\pi\big(\psi_1(s^\star)+\sigma^2\big)}}.
}
]

Let
[
A=\psi_1(s^\star)+\sigma^2,\quad
B=\psi_2(s^\star),\quad
C=\psi_3(s^\star),\ldots
]
Then the first correction factor is the standard saddle/Laplace coefficient
[
\boxed{
1+\frac{C}{8A^2}-\frac{5B^2}{24A^3}+O(A^{-2}),
}
]
after contour deformation to steepest descent.

For implementation, the exact MB integral plus Newton-solved saddle is usable, but the differentiated fixed-order GHQ route below is much cleaner for higher derivatives.

### 15.3 Differentiated GHQ: exact derivatives of the quadrature approximation

For any smooth (g),
[
\mathbb E[g(\mu+\sigma Z)]
\approx
\frac{1}{\sqrt\pi}\sum_{m=1}^M \omega_m, g(t_m),
\qquad
t_m = \mu + \sqrt2,\sigma x_m,
]
where ((x_m,\omega_m)) are Hermite nodes and weights.

Because (t_m) is affine in ((\mu,\sigma)), all mixed derivatives are exact for the chosen quadrature rule:
[
\boxed{
\partial_\mu^a \partial_\sigma^b
\mathbb E[g(\mu+\sigma Z)]
\approx
\frac{(\sqrt2)^b}{\sqrt\pi}
\sum_{m=1}^M
\omega_m, x_m^b, g^{(a+b)}(t_m).
}
]

For CLogLog,
[
g(t)=1-e^{-e^t}.
]
Its first four derivatives are
[
g'(t)= e^{t-e^t},
]
[
g''(t)= (1-e^t)e^{t-e^t},
]
[
g^{(3)}(t)= (e^{2t}-3e^t+1)e^{t-e^t},
]
[
g^{(4)}(t)= (-e^{3t}+6e^{2t}-7e^t+1)e^{t-e^t}.
]

So, for example,
[
\partial_\mu L
\approx
\frac1{\sqrt\pi}\sum_m \omega_m g'(t_m),
\qquad
\partial_\sigma L
\approx
\frac{\sqrt2}{\sqrt\pi}\sum_m \omega_m x_m g'(t_m),
]
[
\partial_{\mu\mu}L
\approx
\frac1{\sqrt\pi}\sum_m \omega_m g''(t_m),
\qquad
\partial_{\mu\sigma}L
\approx
\frac{\sqrt2}{\sqrt\pi}\sum_m \omega_m x_m g''(t_m),
]
[
\partial_{\sigma\sigma}L
\approx
\frac{2}{\sqrt\pi}\sum_m \omega_m x_m^2 g''(t_m),
]
and similarly up to fourth order using the general rule.

This is the implementable answer to D12 if you want exact derivatives of the chosen quadrature approximation rather than symbolic differentiation through adaptive quadrature.

---

## 16. Final corrections / confirmations

### 16.1 The (0.5) in the second mode response

Because (C[\beta_i]\beta_j=C[\beta_j]\beta_i),
[
\beta_{ij}
==========

-H^{-1}\Big(g_{ij}+B_i\beta_j+B_j\beta_i+C[\beta_i]\beta_j\Big)
]
is identical to
[
\beta_{ij}
==========

-H^{-1}\Big(g_{ij}+B_i\beta_j+B_j\beta_i+\tfrac12(C[\beta_i]\beta_j+C[\beta_j]\beta_i)\Big).
]
So the pseudocode and the formula are consistent. The symmetrized form is numerically safer.

### 16.2 Firth and the general derivative split

Yes: if the main LAML machinery uses observed information while the Firth term uses Fisher information, you need two separate curvature providers:

* observed (C[u],Q[u,v],M_i[u]) for the LAML terms,
* Fisher (C_F[u],Q_F[u,v],M_{F,i}[u]) for the Firth correction.

This is exactly the same abstract machinery, just with a different per-observation curvature oracle. Wood’s framework is built around that kind of separation between generic outer calculus and model-specific derivative blocks, and Firth’s correction is the classical Jeffreys-information adjustment. ([School of Mathematics][1])


[1]: https://webhomes.maths.ed.ac.uk/~swood34/gsm.pdf?utm_source=chatgpt.com "Smoothing Parameter and Model Selection for General ..."
[2]: https://academic.oup.com/biomet/article/80/1/27/228364?utm_source=chatgpt.com "Bias reduction of maximum likelihood estimates | Biometrika"
[3]: https://proceedings.mlr.press/v238/auddy24a/auddy24a.pdf?utm_source=chatgpt.com "Approximate Leave-one-out Cross Validation for Regression ..."

