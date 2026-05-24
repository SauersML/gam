# Identifiability Theory for the Composition Engine

This note gives a formal identifiability theory for the `gamfit` composition
engine.  The purpose is not to claim that every non-convex latent model has a
unique numerical optimum.  That statement is false.  The purpose is to state
the exact conditions under which the composed beta/latent/rho objective has one
statistical solution after the only legitimate equivalences have been quotiented
out, and to separate those conditions from the empirical recipe validated in
`auto_exp_38` and `auto_exp_54`.

The setting is a finite sample of size \(N\).  A latent block is
\(\psi=(\psi_1,\ldots,\psi_N)\in\mathbb R^{N\times d}\).  A smooth decoder
\(\Phi:\mathbb R^{N\times d}\to\mathbb R^{N\times p}\) produces the design
matrix \(X_\psi=\Phi(\psi)\).  The inner coefficients are
\(\beta\in\mathbb R^p\).  The registry
\(\mathcal P=\{P_1,\ldots,P_K\}\) contains analytic penalties with tier labels,
so a penalty may act on \(\beta\), on \(\psi\), on assignment variables inside
\(\psi\), on a strength \(\rho\), or on cross-tier combinations.  Write
\(\lambda_k=\exp(\rho_k)\).

For fixed \((\beta,\psi,\rho)\), define the negative joint REML/Laplace
evidence

$$
\mathcal V(\beta,\psi,\rho;y)
=
-\ell(y;X_\psi\beta)
+ \frac12\sum_{k=1}^K \lambda_k P_k(\beta,\psi)
+ A(\rho)
+ \frac12 \log |H_{\beta\beta}(\beta,\psi,\rho)|_+
- \frac12\log |S_\rho|_+
+ C,
$$

where \(H_{\beta\beta}\) is the observed inner Hessian on the active
coefficient tangent space, \(S_\rho=\sum_k\lambda_k S_k\) is the corresponding
coefficient penalty operator, \(A(\rho)\) contains normalized prior constants
such as the ARD \(-\frac12 N_{\mathrm{eff},j}\log\alpha_j\) terms, and
\(|\cdot|_+\) denotes the pseudo-determinant after exact null-space and active
constraint projection.  This is the sign convention used below: the MAP/REML
estimate minimizes \(\mathcal V\).

## Definitions

**Definition 1 (observational equivalence).**  Two triples
\(\theta=(\beta,\psi,\rho)\) and \(\theta'=(\beta',\psi',\rho')\) are
observationally equivalent, written \(\theta\equiv_y\theta'\), if

$$
X_\psi\beta=X_{\psi'}\beta',
\qquad
\mathcal V(\theta;y)=\mathcal V(\theta';y),
$$

and the active coefficient tangent spaces used in the REML log-determinants
are isomorphic under the same reparameterization.  This last clause rules out
declaring two numerically equal fitted means equivalent when one of them has an
extra unpenalized null direction.

**Definition 2 (structural symmetry group).**  Let \(\mathcal G\) be the group
of analytic transformations \(g:\Theta\to\Theta\) that preserve fitted means,
penalty tiers, active tangent spaces, and normalized evidence for every
possible observation \(y\):

$$
X_{g_\psi\psi}g_\beta\beta = X_\psi\beta,
\qquad
\mathcal V(g\theta;y)=\mathcal V(\theta;y).
$$

The quotient parameter is \([\theta]=\mathcal G\theta\).  For latent coordinate
models the candidate generators are rotations
\(\psi\mapsto\psi Q\), scale transformations
\(\psi_j\mapsto c_j\psi_j\) paired with decoder rescaling, and SAE atom
permutations \(k\mapsto\pi(k)\).

**Definition 3 (identifiable composition).**  The composition is identifiable
at \(y\) if the set of global minimizers of \(\mathcal V(\cdot;y)\) is one
orbit in \(\Theta/\mathcal G\).  It is strictly identifiable if, additionally,
the quotient Hessian at that orbit is positive definite.

The quotient Hessian is the restriction of \(\nabla^2\mathcal V\) to any
linear complement of the tangent space \(T_\theta(\mathcal G\theta)\).  Its
positive definiteness is independent of the complement because
\(\mathcal V\) is constant on group orbits.

## Main Identifiability Theorem

**Theorem 1 (composition-engine identifiability).**  Let
\(\Theta\subset\mathbb R^m\) be the finite-dimensional beta/latent/rho
parameter manifold after rank-null constraints and active inequality
constraints have been projected out.  Assume \(\Phi\), the likelihood, and all
registry penalties are real analytic on \(\Theta\), and assume the REML
Hessian and pseudo-determinants are evaluated on the same active tangent space
used by the inner solver.  The MAP estimate
\((\beta^\star,\psi^\star,\rho^\star)\) under joint REML evidence is unique up
to the equivalence class \(\Theta/\mathcal G\) if and only if the following
five conditions hold:

1. **Well-posed evidence.**  \(\mathcal V(\cdot;y)\) is proper and attains its
   minimum on a compact sublevel set, and all active REML Hessians are positive
   definite on their coefficient tangent spaces.
2. **Decoder separation modulo symmetries.**  If two parameter triples have
   the same fitted mean and the same normalized registry value for the
   observation \(y\), then they differ by an element of the structural symmetry
   group \(\mathcal G\).
3. **No residual gauge stabilizer.**  The only rotations, scale changes, and
   atom permutations that preserve the likelihood, the decoder, the normalized
   penalty registry, and the active REML tangent space at a minimizer are the
   elements included in the stated quotient.  Equivalently, the stabilizer of
   each minimizing orbit is exactly the allowed residual group.
4. **Normalized scale evidence.**  Every latent axis whose scale can be traded
   against a decoder or strength parameter has a normalized prior contribution
   with nonzero effective sample size, in particular the ARD term
   \(-\frac12N_{\mathrm{eff},j}\log\alpha_j\) for any learnable latent
   precision \(\alpha_j\), and there is no exactly canceling normalized decoder
   scale term on the same orbit.
5. **Strict quotient curvature.**  On the quotient manifold
   \(\Theta/\mathcal G\), the induced objective has a single stationary global
   minimum and its Hessian is positive definite there.

Under these conditions the representative \((\beta^\star,\psi^\star,\rho^\star)\)
may change with the numerical gauge convention, but every representative has
the same fitted mean, active mechanism count, selected latent axes, and REML
evidence.  If any of the five conditions fails, either the MAP is not
well-defined, or there exist at least two inequivalent global MAP triples, or
there is a nontrivial continuous or discrete flat direction not included in the
quotient.

**Proof.**  Necessity is direct.  If condition 1 fails, a minimizer may not
exist or the Laplace/REML objective is not the stated finite-dimensional
criterion.  If condition 2 fails, two observationally indistinguishable but
non-group-related triples have the same value and produce two quotient points.
If condition 3 fails, the theorem has omitted a true symmetry, so uniqueness
cannot hold for the stated quotient.  If condition 4 fails, Lemma 2 constructs
an exact scale orbit with unchanged likelihood and unchanged unnormalized
penalty value.  If condition 5 fails, the quotient objective either has two
global stationary minima or a flat Hessian direction through a minimizer, so
strict uniqueness on \(\Theta/\mathcal G\) fails.

For sufficiency, condition 1 gives at least one global minimizer.  Conditions
2 and 3 say that equality of fitted mean and registry value at any other
global minimizer can only move inside the declared quotient.  Condition 4
removes the latent scale orbits that are otherwise invisible to the fitted
mean.  Finally condition 5 excludes duplicate quotient minima not generated by
an exact structural symmetry.  Therefore the minimizer set is exactly one
\(\mathcal G\)-orbit, with positive quotient curvature.  \(\square\)

The theorem deliberately places the hard non-convex issue in condition 5.  The
lemmas below identify the concrete registry terms that make conditions 3 and 4
checkable for the composition-engine recipe.

## Symmetry-Breaking Lemmas

**Lemma 1 (auxiliary conditional gauge and rotations).**  Suppose the latent
block has a rotation symmetry
\(\psi\mapsto\psi Q\), \(Q\in O(d)\), in the decoder-likelihood pair.  Add an
`IvaeRidgeMeanGauge` or auxiliary-conditional Gaussian prior

$$
P_{\mathrm{aux}}(\psi)
=
\sum_{i=1}^N(\psi_i-m(u_i))^\top \Lambda(u_i)(\psi_i-m(u_i)),
$$

with \(\Lambda(u_i)\succeq0\).  The auxiliary prior breaks the rotation
symmetry down to

$$
\mathcal H_{\mathrm{aux}}
=
\{Q\in O(d): Q^\top\Lambda(u_i)Q=\Lambda(u_i),\; Q^\top m(u_i)=m(u_i)
\text{ for all } i\}.
$$

In particular, it breaks all rotations on the anchored subspace if and only if
the simultaneous orthogonal commutant of
\(\{\Lambda(u_i)\}_{i=1}^N\) and the fixed-vector constraints
\(\{m(u_i)\}_{i=1}^N\) is trivial on that subspace.

**Proof.**  Under the transformation \(\psi_i\mapsto \psi_iQ\), the prior
value becomes

$$
\sum_i(\psi_iQ-m_i)^\top\Lambda_i(\psi_iQ-m_i).
$$

For this to equal the original quadratic form for all possible \(\psi_i\), the
quadratic, linear, and constant coefficients in \(\psi_i\) must match.  The
quadratic coefficients match exactly when \(Q^\top\Lambda_iQ=\Lambda_i\).  The
linear coefficients match exactly when
\(Q^\top\Lambda_i m_i=\Lambda_i m_i\).  If \(\Lambda_i\) is positive definite
on the anchored coordinates this is equivalent to \(Q^\top m_i=m_i\); with
semidefinite \(\Lambda_i\) the statement holds on the quotient by the
null-space.  Thus the surviving rotations are precisely
\(\mathcal H_{\mathrm{aux}}\).  The group is trivial exactly when the
commutant has no non-identity element on the anchored span.  \(\square\)

This is the finite-sample form of the nonlinear-ICA/iVAE identifiability
condition: the auxiliary variable must change the conditional latent law in
enough independent ways to identify the coordinates.  Khemakhem et al. (2020)
prove the population version for conditional exponential-family priors; here
the registry term supplies the same role in the REML objective.

**Lemma 2 (ARD normalizer and scale).**  Consider one latent axis with decoder
column \(W_j\), coordinate vector \(z_j=\psi_{\cdot j}\), and learnable ARD
precision \(\alpha_j>0\).  Suppose the likelihood is invariant under

$$
z_j\mapsto c z_j,\qquad W_j\mapsto W_j/c,\qquad c>0.
$$

If the registry contains only the unnormalized quadratic
\(\frac12\alpha_j\|z_j\|^2\), then the enlarged transformation
\(\alpha_j\mapsto \alpha_j/c^2\) is an exact flat scale symmetry.  If the
registry contains the normalized ARD contribution

$$
\frac12\alpha_j\|z_j\|^2
-\frac12N_{\mathrm{eff},j}\log\alpha_j,
$$

with \(N_{\mathrm{eff},j}>0\), then that scale symmetry is broken, except for
any explicitly quotiented discrete sign change or any exactly canceling
normalized decoder-scale term.

**Proof.**  The fitted mean is unchanged by \(z_j\mapsto cz_j\),
\(W_j\mapsto W_j/c\).  In the unnormalized case,

$$
\frac12(\alpha_j/c^2)\|cz_j\|^2
=
\frac12\alpha_j\|z_j\|^2,
$$

so the likelihood and the ARD penalty are both constant along \(c\).  The
scale is not identifiable.

With the normalized term, the transformed negative log prior is

$$
\begin{aligned}
&\frac12(\alpha_j/c^2)\|cz_j\|^2
-\frac12N_{\mathrm{eff},j}\log(\alpha_j/c^2) \\
&\qquad =
\frac12\alpha_j\|z_j\|^2
-\frac12N_{\mathrm{eff},j}\log\alpha_j
+ N_{\mathrm{eff},j}\log c.
\end{aligned}
$$

The derivative with respect to \(\log c\) is \(N_{\mathrm{eff},j}\), which is
nonzero by assumption.  Hence the old scale orbit is no longer a level set of
the evidence.  A remaining flat direction can occur only if another normalized
term contributes exactly \(-N_{\mathrm{eff},j}\log c\) on the same
transformation, or if the transformation is a sign flip already included in
the quotient.  \(\square\)

The long calculation in this proof is the reason the implementation requires
the REML normalizer for ARD and auxiliary priors.  Without it, ARD can look
like dimension selection while leaving the \(\alpha/W\) scale gauge intact.

**Lemma 3 (IBP ordering and atom permutations).**  Let \(Z\in\{0,1\}^{N\times K}\)
be the SAE-manifold atom indicator matrix and let
\(\bar n_k=N^{-1}\sum_i Z_{ik}\).  An exchangeable finite IBP prior identifies
atoms only up to permutation.  If the registry uses the canonical
left-ordered IBP representative, ordering atoms by decreasing expected
occupancy, and if the true occupancies satisfy
\(\mu_1>\mu_2>\cdots>\mu_K\) with
\(\Delta=\min_{k<K}(\mu_k-\mu_{k+1})>0\), then the probability that empirical
ordering disagrees with population ordering is at most

$$
\Pr(\exists k:\bar n_k\le \bar n_{k+1})
\le
(K-1)\exp(-N\Delta^2/2).
$$

Thus the IBP indicator penalty breaks atom permutation in expectation at rate
\(O(K\exp(-N\Delta^2/2))\).  Tied expected occupancies remain identifiable
only up to permutation within each tie block.

**Proof.**  For adjacent atoms define
\(D_k=\bar n_k-\bar n_{k+1}\).  Since \(Z_{ik}-Z_{i,k+1}\in[-1,1]\) and
\(\mathbb E D_k=\mu_k-\mu_{k+1}\ge\Delta\), Hoeffding's inequality gives

$$
\Pr(D_k\le0)
=
\Pr(D_k-\mathbb E D_k\le -\mathbb E D_k)
\le
\exp(-N\Delta^2/2).
$$

A union bound over \(K-1\) adjacent inversions gives the result.  If
\(\Delta=0\), the bound does not decay and the prior is genuinely symmetric
inside the tied block.  \(\square\)

**Lemma 4 (supervised block transfers a gauge to a free block).**  Partition
\(\psi=(\psi_S,\psi_F)\), where \(\psi_S\in\mathbb R^{N\times r}\) is anchored
by supervised auxiliary targets and \(\psi_F\in\mathbb R^{N\times(d-r)}\) is
free.  Suppose Lemma 1 makes the supervised stabilizer trivial on
\(\psi_S\), and suppose the registry includes a between-block orthogonality
term

$$
P_{\perp}(\psi_S,\psi_F)
=
\|\psi_S^\top W \psi_F\|_F^2
$$

with \(W\succ0\) on the row span used by the fit.  Then no rotation can mix
\(\psi_S\) and \(\psi_F\) without increasing the objective, unless the mixed
direction lies in the zero row span of both blocks.  The remaining symmetry is
only a within-free-block rotation, later broken by an isometry, ARD after
gauge fixing, or an explicit quotient.

**Proof.**  A rotation mixing supervised and free coordinates has block form
\(\begin{psmallmatrix}A&B\\ C&D\end{psmallmatrix}\) with either \(B\ne0\) or
\(C\ne0\).  Because the supervised prior has trivial stabilizer, \(A\) cannot
absorb \(B\psi_F\) while preserving all supervised mean and precision
constraints.  Independently, the first variation of
\(\|\psi_S^\top W\psi_F\|_F^2\) in a mixed direction contains terms
\(\psi_S^\top W\psi_SB\) and \(C^\top\psi_F^\top W\psi_F\).  These vanish for
all mixed directions only if the relevant block Gram matrix is singular on the
mixed subspace.  Since \(W\succ0\) on the row span, singularity means the mixed
coordinate is zero in the fitted row span.  Such a coordinate carries no
statistical content and belongs in the quotient or should be removed as dead
code.  \(\square\)

**Lemma 5 (joint supervised, ARD, and IBP identifiability).**  For an
SAE-manifold or latent-coordinate composition, assume: the supervised
auxiliary prior satisfies Lemma 1; every active latent scale satisfies Lemma 2;
the atom indicators satisfy Lemma 3 with tie blocks explicitly quotiented; and
the quotient Hessian is positive definite at the global REML minimum.  Then
the composition is strictly identifiable up to signs of symmetric axes,
within-tie atom permutations, and unused zero-measure latent directions.

**Proof.**  Lemma 1 removes rotations on the supervised block.  Lemma 4
prevents that block from being mixed with free coordinates unless the mixed
coordinates are statistically unused.  Lemma 2 removes continuous latent scale
orbits for active axes.  Lemma 3 replaces arbitrary atom labels by a canonical
ordering with exponentially small finite-sample inversion probability, leaving
only tie-block permutations.  After those transformations have been removed or
declared in the quotient, the positive quotient Hessian gives local strictness
and condition 5 of Theorem 1 gives global uniqueness.  \(\square\)

**Corollary 1 (why `IvaeRidgeMeanGauge` plus supervised aux works).**  If the
supervised auxiliary variables span the intended concept rank, the conditional
ridge means or precisions are full rank on that span, and the REML objective
includes the normalized auxiliary/ARD terms, then `IvaeRidgeMeanGauge` breaks
rotation, ARD breaks active-axis scale, and ordered IBP breaks atom relabeling
up to tied or unused atoms.  This is exactly the regime tested by
`auto_exp_38`: HSV supervision fixed the three-dimensional gauge, and the
free companion block became interpretable rather than arbitrarily rotated into
HSV.

## Inner Penalized-Newton Convergence

The composition engine also needs a solver theorem.  The inner loop minimizes,
for fixed \(\rho\) and current design snapshot,

$$
F(\theta)
=
-\ell(y;\eta(\theta))
+ \frac12\sum_k \lambda_k P_k(\theta),
\qquad
\theta=(\beta,\psi_{\mathrm{inner}}),
$$

where \(\eta(\theta)\) may be nonlinear because \(\Phi(\psi)\beta\) depends on
\(\psi\).  Let \(g_t=\nabla F(\theta_t)\).  The damped penalized-Newton step
computes a symmetric model Hessian \(B_t\) satisfying
\(B_t\succeq \mu I\) after active-space projection or trust-region
regularization, solves \(B_tp_t=-g_t\) up to a forcing error
\(\|B_tp_t+g_t\|\le \kappa\min(1,\|g_t\|)\|g_t\|\), and chooses
\(\alpha_t\in(0,1]\) by Armijo decrease:

$$
F(\theta_t+\alpha_t p_t)
\le
F(\theta_t)+c\alpha_t g_t^\top p_t,
\qquad c\in(0,1).
$$

**Theorem 2 (global convergence of the inner penalized-Newton loop).**  Assume
\(F\) is continuously differentiable, bounded below, and has Lipschitz
gradient on the initial sublevel set
\(\{\theta:F(\theta)\le F(\theta_0)\}\).  Assume that sublevel set is compact
after the declared gauge quotient, and that the projected model Hessians are
uniformly bounded above and below:
\(\mu I\preceq B_t\preceq L_B I\).  Then every accumulation point of the
damped penalized-Newton iterates is a first-order critical point of \(F\) on
the active tangent space.  If an accumulation point has positive definite true
Hessian on that tangent space and the model Hessian converges to the true
Hessian, then the full Newton step is eventually accepted and convergence to
that point is quadratic for exact solves, superlinear for forcing terms
approaching zero, and linear otherwise.

**Proof.**  The inexact Newton solve and the spectral bounds imply a descent
bound.  Since \(B_tp_t=-g_t+r_t\) with
\(\|r_t\|\le \kappa\min(1,\|g_t\|)\|g_t\|\), choosing the standard forcing
\(\kappa<1\) gives

$$
g_t^\top p_t
=
-p_t^\top B_t p_t+p_t^\top r_t
\le
-\mu\|p_t\|^2+\|p_t\|\|r_t\|.
$$

The same equation and \(B_t\preceq L_BI\) give
\(\|p_t\|\ge (1-\kappa)\|g_t\|/L_B\) whenever \(\|g_t\|\le1\), and an analogous
bounded-gradient version outside that ball.  Hence \(p_t\) is gradient-related:
there is \(\gamma>0\) such that

$$
g_t^\top p_t\le -\gamma\|g_t\|^2
$$

whenever \(g_t\ne0\).  Lipschitz continuity of \(\nabla F\) gives

$$
F(\theta_t+\alpha p_t)
\le
F(\theta_t)+\alpha g_t^\top p_t
+\frac12L\alpha^2\|p_t\|^2.
$$

For sufficiently small \(\alpha\), the quadratic term is dominated by
\((1-c)\alpha(-g_t^\top p_t)\), so Armijo backtracking terminates.  The
accepted step therefore satisfies

$$
F(\theta_t)-F(\theta_{t+1})
\ge
c\alpha_t(-g_t^\top p_t).
$$

The line-search lower bound implied by the previous display and the uniform
Hessian bounds yields
\(F(\theta_t)-F(\theta_{t+1})\ge a\|g_t\|^2\) for some \(a>0\) along any
subsequence bounded away from criticality.  Since \(F\) is bounded below, the
decreases are summable; therefore \(\liminf_t\|g_t\|=0\).  Compactness gives
an accumulation point.  If an accumulation point had nonzero projected
gradient, continuity would keep \(\|g_t\|\) bounded away from zero in a
neighborhood, forcing a fixed positive decrease on every visit, contradicting
boundedness below.  Thus every accumulation point is critical.

For the local rate, positive definiteness of the true Hessian and
\(B_t\to\nabla^2F(\theta^\star)\) reduce the method to the classical Newton
recursion
\(\|e_{t+1}\|\le C\|e_t\|^2+\|B_t^{-1}r_t\|\).  Exact solves give the quadratic
term.  Vanishing forcing terms give superlinear convergence; fixed forcing
gives linear convergence.  Non-convexity is harmless for this conclusion
because the theorem claims convergence to a critical point, not to the global
minimum.  \(\square\)

## Empirical Validation Map

`auto_exp_38` is the principal positive validation.  HSV supervision supplied
the auxiliary conditional gauge, recovering
\(R^2(\mathrm{hue},\mathrm{sat},\mathrm{val})=(0.700,0.657,0.719)\).  The free
companion block remained mostly orthogonal to HSV and exposed name-semantic
axes.  This validates Lemma 1 and Lemma 4 in the intended LLM setting.

`auto_exp_54` is the target-invariance check.  The same supervised gauge-fix
recipe recovered name-semantic targets with cross-validated
\(R^2=0.763\) for modifier count, \(0.733\) for monoword, and \(0.620\) for
template sigma.  This supports the theorem's claim that the recipe is not
HSV-specific; it is a rank-and-auxiliary-variation condition.

`auto_exp_47` explains why topology alone is insufficient.  A Circle latent
failed on raw PCs even though an oracle \(PC2+PC4\) plane had circular
correlation about \(-0.720\) with hue.  In theorem language, the cyclic
topology did not satisfy decoder separation modulo the semantic gauge.

`auto_exp_49` and `auto_exp_50` validate Lemma 2 by failure.  ARD over PCs,
even with improved scale normalization, followed reconstruction variance and
did not discover the semantic cycle.  The normalized ARD term is necessary for
scale identifiability, but it is not sufficient for concept identifiability
without the auxiliary conditional gauge.

These empirical results are consistent with GPFA and mGPLVM practice.  GPFA
identifies smooth latent trajectories only after choosing a latent dimension
and covariance structure; rotations of factor loadings remain a standard
factor-model issue.  mGPLVM shows that non-Euclidean topology such as a circle
or torus can recover neural variables like head direction when the likelihood
and topology align, but `auto_exp_47` shows that topology cannot name a
sub-dominant semantic factor by itself.  The SAE literature reaches the same
warning from another direction: sparse autoencoders give useful feature
dictionaries, but feature labels and atom counts require additional functional
or coding assumptions rather than reconstruction sparsity alone.

## Limitations and Open Problems

First, Theorem 1 is a finite-sample quotient theorem, not a guarantee that an
arbitrary optimizer finds the global quotient minimum.  Theorem 2 gives
critical-point convergence for the inner damped Newton loop; outer REML
multimodality remains a model-comparison problem.

Second, the IBP result is an ordered-occupancy statement.  If two atoms have
equal expected occupancy or identical decoder action, the model is identifiable
only up to their permutation.  This is a feature, not a defect: relabeling
indistinguishable atoms is not statistical information.

Third, ARD is a scale and dimension-pressure device after the chart is fixed.
It is not a semantic concept finder.  The strongest open conjecture is that,
for sub-dominant cyclic concepts in high-dimensional activation clouds, no
reconstruction-only objective with exchangeable ARD and topology priors can
consistently recover the semantic cycle without an auxiliary variable whose
conditional law varies along that concept.

Fourth, the theory assumes analytic or smoothed penalties.  Exact nonsmooth
\(\ell_1\), hard top-\(k\), and discontinuous assignment updates require a
Clarke-critical or variational-analysis version of Theorem 2.  The expected
statement should hold for prox-regular penalties, but the proof is not in this
note.

Fifth, the quotient Hessian condition is local.  It can be checked numerically
by active-space Hessian spectra, but a useful symbolic criterion for complex
SAE-manifold decoders is still missing.

## References

- Yu, Cunningham, Santhanam, Ryu, Shenoy, and Sahani (2009),
  ["Gaussian-process factor analysis for low-dimensional single-trial analysis
  of neural population activity"](https://pmc.ncbi.nlm.nih.gov/articles/PMC2712272/).
- Khemakhem, Kingma, Monti, and Hyvarinen (2020),
  ["Variational Autoencoders and Nonlinear ICA: A Unifying
  Framework"](https://proceedings.mlr.press/v108/khemakhem20a.html).
- Jensen, Kao, Tripodi, and Hennequin (2020), ["Manifold GPLVMs for
  discovering non-Euclidean latent structure in neural
  data"](https://papers.nips.cc/paper_files/paper/2020/hash/fedc604da8b0f9af74b6cfc0fab2163c-Abstract.html).
- Braun, Taylor, Goldowsky-Dill, and Sharkey (2024), ["Identifying
  Functionally Important Features with End-to-End Sparse Dictionary
  Learning"](https://arxiv.org/abs/2405.12241).
- Leask, Bussmann, Pearce, Bloom, Tigges, Al Moubayed, Sharkey, and Nanda
  (2025), ["Sparse Autoencoders Do Not Find Canonical Units of
  Analysis"](https://arxiv.org/abs/2502.04878).
