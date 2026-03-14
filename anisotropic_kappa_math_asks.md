# Anisotropic Length-Scale for Radial Kernels: Math Asks

## Context for the Math Team

We have a GAM (generalized additive model) engine that fits penalized regression
splines using radial basis functions — specifically Matérn and hybrid Duchon
kernels. The engine currently supports **isotropic** kernels: every kernel
evaluation depends on the scalar Euclidean distance r = ‖x - x'‖ between two
points. There is a single length-scale parameter ℓ (equivalently κ = 1/ℓ) that
uniformly scales this distance.

We want to extend this to **anisotropic** (per-axis) length-scales for
multi-dimensional spatial terms. The motivating application is polygenic score
calibration over d = 4–16 principal components of genetic ancestry, where
different PC axes carry very different amounts of signal.


## What We Currently Have

### Kernels

**Matérn (half-integer ν):**

    φ(r; κ) = P_ν(a) exp(−a),    a = √(2ν) · κ · r

where P_ν is a polynomial of degree ν − 1/2. We support ν ∈ {1/2, 3/2, 5/2, 7/2, 9/2}.

**Hybrid Duchon (orders p, s):**

The kernel is defined via its isotropic spectral density:

    K̂(ω; κ) ∝ 1 / (|ω|^(2p) · (κ² + |ω|²)^s)

In the spatial domain this decomposes via partial fractions into polyharmonic
blocks Φ_m(r) (pure power/log functions of r) plus Matérn–Bessel blocks
M_n(r; κ) = c · r^ν · K_ν(κr). The full kernel scales as:

    φ(r; κ) = κ^δ · H(κr),    δ = d − 2p − 2s

### Three collocation operators

From the kernel and its radial derivatives, we build three n × k matrices
(data-to-knots), where k is the number of knot centers:

- **D₀[i,j] = φ(r_ij)** — kernel value ("mass")
- **D₁[(i,a), j] = q(r_ij) · (x_{i,a} − c_{j,a})** — gradient, where q = φ'/r
- **D₂[i,j] = Δφ(r_ij) = φ'' + (d−1)·q** — Laplacian ("stiffness")

Penalty matrices are Gram products: S_m = D_m^T D_m, each getting its own
smoothing parameter λ_m selected by REML.

### Scalar κ optimization

We parameterize ψ = log(κ) and jointly optimize θ = [ρ, ψ] where ρ are
log-smoothing parameters and ψ is the log-kappa for each spatial term. The REML
objective, gradient, and Hessian w.r.t. ψ are computed analytically using:

    ∂φ/∂ψ = δ·φ + r·φ'        (Duchon scaling law)
    ∂²φ/∂ψ² = δ²·φ + (2δ+1)·r·φ' + r²·φ''

For Matérn, the equivalent chain through a = √(2ν)·κ·r:

    ∂φ/∂ψ = a·(P'_ν(a) − P_ν(a))·exp(−a)

The same scaling law extends to the operators q and Δφ (with exponent δ+2), and
penalty derivatives follow from the Gram product rule:

    S_ψ = D_ψ^T D + D^T D_ψ
    S_ψψ = D_ψψ^T D + 2·D_ψ^T D_ψ + D^T D_ψψ

All of this is fully analytic (no finite differences). The optimizer is
Newton trust-region on the full [ρ, ψ] space.


## What We Want: Per-Axis κ

Replace the scalar distance with an anisotropic one:

    r_iso = ‖Δx‖ = √(Σ_a Δx_a²)

    r_aniso = ‖Λ Δx‖ = √(Σ_a κ_a² Δx_a²)

where Λ = diag(κ₁, ..., κ_d). The kernel is still φ(r_aniso), using the same
radial function. Each axis gets its own ψ_a = log(κ_a).


## Math Asks

### Ask 1: Derivative chain for per-axis ψ_a

We need ∂r/∂ψ_a and ∂²r/∂ψ_a∂ψ_b to chain through the existing radial
derivatives φ'(r), φ''(r).

**Our current derivation (please verify):**

    ∂r/∂ψ_a = κ_a² Δx_a² / r

    ∂²r/∂ψ_a² = κ_a² Δx_a² (r² − κ_a² Δx_a²) / r³

    ∂²r/∂ψ_a∂ψ_b = −κ_a² Δx_a² · κ_b² Δx_b² / r³    (a ≠ b)

Then:

    ∂φ/∂ψ_a = φ'(r) · ∂r/∂ψ_a

    ∂²φ/∂ψ_a∂ψ_b = φ''(r) · (∂r/∂ψ_a)(∂r/∂ψ_b) + φ'(r) · ∂²r/∂ψ_a∂ψ_b

**Question:** Are these correct? Are there simplifications we're missing?
In particular, is there a compact matrix form that avoids computing all d²
cross terms explicitly?


### Ask 2: Collision limit (r → 0)

When a data point coincides with a knot center, r → 0 and the above expressions
have 0/0 forms. For the isotropic case we handle this via L'Hôpital / Taylor
expansion at r = 0.

**Question:** What are the correct collision limits for the anisotropic
derivatives? Specifically:

- lim_{r→0} ∂r/∂ψ_a = ?
- lim_{r→0} ∂φ/∂ψ_a = ?
- lim_{r→0} (∂/∂ψ_a)[φ'(r)/r] = ?  (this enters the gradient operator D₁)
- lim_{r→0} (∂/∂ψ_a)[Δφ(r)] = ?  (this enters the Laplacian operator D₂)

For the isotropic case, φ'(r)/r → φ''(0) as r → 0, and all ψ-derivatives at
collision reduce to expressions involving φ''(0) and higher even derivatives.
We need the anisotropic analogues.


### Ask 3: Scaling law for hybrid Duchon under anisotropy

The isotropic hybrid Duchon kernel has a clean scaling law:

    φ(r; κ) = κ^δ · H(κr),    δ = d − 2p − 2s

which gives closed-form ψ-derivatives without computing through the
partial-fraction expansion:

    φ_ψ = δ·φ + r·φ'
    φ_ψψ = δ²·φ + (2δ+1)·r·φ' + r²·φ''

**Question:** Does a comparable scaling identity exist for the anisotropic case?

In the isotropic case, the spectral density is:

    1 / (|ω|^(2p) · (κ² + |ω|²)^s)

With per-axis κ, a natural spectral generalization would be:

    1 / (|ω|^(2p) · (Σ_a κ_a² + |ω|²)^s)     [Option A: scalar mass term]

or:

    1 / ((Σ_a ω_a²)^p · (Σ_a (κ_a² + ω_a²))^s)     [Option B: per-axis mass]

or something else entirely. Option A just replaces κ² with ‖κ‖² in the mass
term — this doesn't actually give per-axis behavior, it's still isotropic
with a different effective κ. Option B breaks the isotropic structure of the
spectral density.

**The core question is:** what is the "right" anisotropic generalization of
the hybrid Duchon spectrum? Is it:

    1 / ((Σ_a κ_a² ω_a²)^p · (1 + Σ_a ω_a²/κ_a²)^s)     [Option C]

or is there a standard construction? The kernel must remain positive definite
(or at least conditionally positive definite with the same polynomial null
space). We need guidance on which spectral form to use and whether the
resulting spatial kernel still has a partial-fraction decomposition that we can
evaluate.


### Ask 4: Operator collocation under anisotropy

The gradient operator D₁ currently uses the isotropic gradient of φ:

    ∇_a φ(r) = φ'(r)/r · Δx_a = q(r) · Δx_a

Under the anisotropic distance r = ‖Λ Δx‖, the chain rule gives:

    ∂φ/∂x_a = φ'(r) · ∂r/∂x_a = φ'(r) · κ_a² Δx_a / r

So the gradient operator becomes:

    ∇_a φ = κ_a² · q(r) · Δx_a

where q = φ'(r)/r as before, but r is now the anisotropic distance.

**Question:** Is this correct? And for the Laplacian:

    Δφ = Σ_a ∂²φ/∂x_a² = Σ_a [κ_a⁴ Δx_a² · φ''(r)/r² + κ_a² · (φ'(r)/r − κ_a² Δx_a² · φ'(r)/r³)]

We think this simplifies to:

    Δφ = φ''(r)/r² · Σ_a κ_a⁴ Δx_a² + φ'(r)/r · (Σ_a κ_a² − Σ_a κ_a⁴ Δx_a²/r²)

but we want confirmation and the collision limit (r → 0).

**Follow-up:** The penalty S₁ = D₁^T D₁ currently penalizes total gradient
energy ‖∇f‖². Under anisotropy, should we still form S₁ from the gradient
∇_a φ = κ_a² · q · Δx_a (which penalizes the gradient in the original
coordinate system, weighted by κ), or from the gradient in the rescaled
coordinate system y_a = κ_a · x_a (which would just be q · Δy_a, identical
to the isotropic formula in y-space)?

The choice matters because it determines whether the penalty has additional
κ-dependence beyond the distance.


### Ask 5: Penalty matrix ψ-derivatives under anisotropy

In the isotropic case, each collocation operator depends on ψ only through r,
so the ψ-derivative of D is obtained by chaining through φ'(r) · ∂r/∂ψ.
The penalty derivative S_ψ = D_ψ^T D + D^T D_ψ follows from the product rule.

Under per-axis ψ, D₁ has an **explicit** κ_a² factor multiplying
q(r) · Δx_a, in addition to the implicit dependence through r. So:

    ∂D₁[(i,a),j]/∂ψ_b = [∂(κ_a²)/∂ψ_b] · q(r) · Δx_a
                        + κ_a² · [∂q(r)/∂ψ_b] · Δx_a

The first term is 2κ_a² · q · Δx_a when b = a, and 0 when b ≠ a.
The second term chains through r: ∂q/∂ψ_b = q'(r) · ∂r/∂ψ_b where
q' = dq/dr = (φ'' − q)/r.

**Question:** Please verify this derivative structure and confirm the second
derivatives (∂²D₁/∂ψ_a∂ψ_b) are correct. The penalty Hessian will have
d × d blocks and we want to make sure we aren't missing terms.

Similarly for D₂ (Laplacian operator): the anisotropic Laplacian itself depends
on κ explicitly (not just through r), so its ψ-derivatives have both explicit
and implicit parts. Please derive these.


### Ask 6: Identifiability and parameter count

With d per-axis log-kappa values, one degree of freedom is redundant with the
overall smoothing parameters λ. Specifically, uniformly scaling all κ_a by a
constant c is equivalent to changing κ → c·κ in the isotropic case — which
just shifts the smoothing parameters.

**Question:** Is there a natural identifiability constraint we should impose?
Options we've considered:

(a) Fix Π κ_a = 1 (geometric mean constraint), optimizing d−1 free parameters
(b) Fix Σ ψ_a = 0 (sum-to-zero on log scale, equivalent to (a))
(c) Fix κ₁ = 1 (anchor one axis), optimizing d−1 free parameters
(d) Don't constrain — let REML sort it out via the λ interaction

Our intuition is that (d) is fine because the λ's and κ's live in different
parts of the model (λ controls penalty weight, κ controls kernel shape) and
REML can distinguish them. But we want confirmation that the joint [ρ, ψ₁, ..., ψ_d]
parameterization doesn't have a ridge/valley in the REML surface that would
cause optimizer difficulties.


### Ask 7: Positive definiteness under anisotropy

For Matérn kernels, the anisotropic version φ(‖Λ(x−x')‖) is known to be
positive definite whenever the isotropic φ(‖·‖) is, because Λ is just a
linear transformation of the input space. This is Schoenberg's theorem.

**Question:** Does the same hold for conditionally positive definite kernels
(Duchon splines with p > 0)? Specifically, if φ is CPD of order m on ℝ^d
with isotropic distance, is φ(‖Λ·‖) still CPD of the same order?

Our belief is yes (since Λ is a bijection, the polynomial null space transforms
accordingly), but we want a clean argument.


### Ask 8: Initialization heuristic

For the isotropic case, we initialize κ from the pairwise distance distribution
of the knot centers. With per-axis κ, we plan to initialize each κ_a from
the marginal distance distribution along axis a.

**Question:** Is there a better initialization? In particular:

- Should we initialize κ_a ∝ 1/σ_a where σ_a is the standard deviation of
  axis a (equivalent to standardizing the inputs)?
- Or κ_a ∝ 1/√(eigenvalue_a) for PCA axes (equivalent to scaling by the
  amount of variance each PC explains)?
- Or something derived from the data-to-knot distance matrix?

We want an initialization that's likely to be in the basin of attraction of the
REML optimum, to minimize outer iterations.


## Summary of Deliverables

1. **Verify** our derivative chain (Ask 1)
2. **Derive** collision limits (Ask 2)
3. **Recommend** the anisotropic Duchon spectral form and whether partial-fraction decomposition survives (Ask 3)
4. **Verify** anisotropic gradient/Laplacian operators and advise on penalty coordinate system (Ask 4)
5. **Verify** operator ψ-derivative structure under anisotropy (Ask 5)
6. **Advise** on identifiability constraints (Ask 6)
7. **Confirm** CPD preservation under anisotropic distance (Ask 7)
8. **Recommend** initialization heuristic (Ask 8)
