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

## Section 1 — REML/LAML criterion-gradient problems

*(self-contained statement: see `reml_criterion_gradients.md`)*

## Section 2 — Statistical-genetics identification and the survival warp gauge

*(self-contained statement: see `statgen_identification.md`)*

## Section 3 — Riemannian-manifold geometry and Matérn penalty enumeration

*(self-contained statement: see `manifold_and_matern.md`)*

---

## What we want back, per problem

For each numbered problem: (1) the exact object or identity, derived; (2) a proof of correctness — including a proof that it reduces to the stated known special case and extends to the failing regime; (3) where relevant, an **independent self-checking identity** (a scalar or matrix equality that two distinct derivations must satisfy) that could serve as a correctness certificate without running code; (4) a clear statement of any regime where the problem is genuinely ill-posed, with the impossibility argument. Pseudocode is welcome as mathematics; source code in any language is neither needed nor wanted.

Prefer one fully-correct general theorem to several special cases. If you see a deeper unifying structure across the three sections (they are all, in different guises, about differentiating a constrained/profiled criterion through a moving subspace), say so and exploit it.
