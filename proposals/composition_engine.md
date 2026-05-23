# gamfit as a Penalized-Likelihood Composition Engine

Status: draft RFC
Audience: gamfit maintainers
Scope: architectural framing + a small, bounded set of new primitives that turn gamfit into a substrate for unsupervised representation learning, principal-manifold estimation, identifiable latent-variable models, and Fisher-Rao-aware geometric fitting, with most of the surface area already implemented.

## 1. Premise

The "GAM library" framing undersells what gamfit owns. Read the source bottom-up rather than top-down and the actual abstraction is visible: a *penalized-likelihood inference engine* with a *marginal-likelihood model selector* (REML / LAML, implemented in `src/solver/reml/`) wrapped around an *implicit-differentiation nested optimizer* (`src/solver/pirls.rs`, `src/solver/reml/runtime.rs::evaluate_unified_with_psi_ext`). Smooth functions of observed covariates with REML-selected wiggliness are the historical use case, but they are *one configuration* of the machine, not the machine itself. The machine accepts:

- a design block whose columns may depend on hyperparameters,
- a penalty block whose Gram matrix may depend on hyperparameters,
- first/second derivatives of both with respect to those hyperparameters,
- a weighted likelihood (working weight matrix, possibly user-supplied),

and it returns the marginal-likelihood-optimal hyperparameters together with the inner-loop coefficients, with derivatives propagated through the inner solution by the implicit function theorem (IFT). Any statistical procedure that can be cast in that shape is something gamfit already fits. Most cross-pollinations people propose from interpretability, neuroscience, representation learning, and modern diffusion-model geometry are recompositions of primitives gamfit already owns. This proposal makes that mapping explicit and then identifies the minimal set of *new* primitives that close the gap.

The pitch is not to widen the API surface. It is to recognize that the existing surface is already very wide, name the missing pieces precisely, and add them without proliferating one-off helpers.

## 2. The mapping

| Cross-domain idea | gamfit primitive | Owner in source |
|---|---|---|
| Smooth decoder `M → ℝ^p` (GP-LVM, principal manifolds, kernel PCA) | multi-output smooth on existing bases | `Smooth` (`gamfit/smooth.py:34`), `Duchon`/`Sphere`/`PeriodicSplineCurve` subclasses, `SmoothTerm` (`src/terms/smooth.rs:320`), `BasisFamily` (`src/terms/basis.rs:238`) |
| Smoothness / wiggliness selection | `S(ρ)` penalty + REML / LAML | `RemlState::evaluate_unified` (`src/solver/reml/runtime.rs:8069`), `laml_*` proof scaffolding (`src/solver/reml.lean:164-230`) |
| Latent coordinate estimation (GP-LVM, autoencoder bottleneck) | ext-coordinate block whose design derivatives reuse the existing kernel-shape derivative machinery | landed in the standard path; see §4(a); pattern is `SpatialLogKappaCoords` (`src/terms/smooth.rs:1774`) and `LatentCoordValues` (`src/terms/latent_coord.rs:561`) |
| Intrinsic-dimension selection (auto-discovery of latent dim) | ARD: one penalty per latent axis, REML drives unused axes to ∞ *given a paired gauge fix* (aux-conditional prior or isometry) | composition of (a) + per-axis `S(ρ)` + paired gauge-fix from (b) or (c) + existing REML; no new primitive |
| Topology choice (S¹, S², torus, Euclidean patch) | basis family + tensor product | `PeriodicSpline1D` (`src/terms/basis.rs:1423`), `SphericalSplineBasisSpec` (`src/terms/basis.rs:2657`), `DuchonBasisSpec` (`src/terms/basis.rs:2786`), `TensorBSplineSpec` (`src/terms/smooth.rs:295`); TDA persistent cohomology suggests which |
| Gauge / canonical coordinate (Riemannian isometry to a reference) | structured penalty pinning the pullback metric toward a reference | needs new — see §4(b) |
| Identifiability via auxiliary variable (iVAE) | conditional prior on the latent block; mechanism sparsity = L¹-like on a sub-block | needs new — see §4(c); reuses `BlockwisePenalty` (`src/terms/smooth.rs:707`) |
| Behavioral vs geometric inner product (Fisher-Rao) | the working weight matrix `W` in PIRLS | already there — `WorkingModel::update`/`update_with_curvature` (`src/solver/pirls.rs:526-529`); `W` is computed inside the working model, so a custom working model that returns Fisher information *is* Fisher-Rao fitting |
| Geodesic-acceleration Newton (Transtrum) | second-order Newton correction in the inner solver | landed behind `WorkingModelPirlsOptions::geodesic_acceleration` (`src/solver/pirls.rs:1289`) with the dense-factor correction block at `src/solver/pirls.rs:5086-5153`; see §4(e) |
| Mechanism / sparse-coding amplitude gating | `by` multiplier on a smooth | `Smooth.by` (`gamfit/smooth.py:74`); already supported |
| Closed-form linear operator penalties (PDE constraints, divergence-free, …) | `OperatorPenaltySpec` | `OperatorPenaltySpec` (`src/terms/basis.rs:2858`), `ClosedFormPenaltyOperator` (`src/terms/closed_form_operator.rs:26`) |

The right column is mostly existing code. The "needs new" rows are §4.

## 3. The impossibility-theorem framing

Hyvärinen and Pajunen (1999) established that nonlinear ICA without auxiliary structure is unidentifiable: an arbitrary nonlinear mixing of independent latents can be undone by infinitely many distinct unmixings, all with the same observed-data likelihood. The same difficulty pervades unsupervised representation learning more broadly: a smooth invertible reparameterization of the latent space leaves the marginal likelihood of the observed data unchanged, so any objective that depends only on the marginal has a continuous symmetry group of equivalent optima. Recovery is possible only when the model class is restricted in a way that breaks that symmetry — auxiliary variables (iVAE, Khemakhem et al. 2020), volume-preservation, sparsity priors, or topology constraints.

gamfit's primitive vocabulary is precisely the language of inductive biases:

- *bases* encode topology priors (a `PeriodicSpline1D` on a latent axis is the inductive bias "this coordinate lives on S¹"; a `Sphere` term is the inductive bias "this coordinate lives on S²");
- *penalties* encode regularity and identifiability priors (a Duchon `S(ρ)` is "smoothness of order m"; an L¹-style penalty on a parameter block is mechanism sparsity; an isometry penalty against a reference metric is a gauge choice);
- *working weights* `W` encode metric priors (output Fisher information turns mean-squared error into Fisher-Rao distance);
- *REML / LAML* chooses the strength of all of the above by marginal-likelihood comparison, with derivatives done correctly via the existing IFT plumbing.

Lean into this. The marketing line for the broader community should not be "gamfit fits GAMs"; it should be "gamfit is a language for specifying inductive biases and a principled selector for their strengths." Every "GAM" is a particular sentence in that language.

## 4. Concrete additions needed

The new code is small. Each addition reuses an existing pattern; none requires inventing new IFT machinery.

### 4(a). `LatentCoord` parameter type

**Problem.** A GP-LVM, principal manifold, or autoencoder bottleneck needs the inner solver to estimate not only the decoder coefficients `β` but also a per-observation latent coordinate `t_n ∈ ℝ^d`. The design matrix `X` is then a function of `t`: each row is the evaluation of the basis at the row's latent coordinate. This is the ext-coordinate dependence pattern, just shifted from a few global length-scale knobs to one (small) vector per observation. Reserve `ψ` for the existing kernel-shape machinery (`SpatialLogKappaCoords`); the per-row latent is always `t`.

**Existing analogue.** `SpatialLogKappaCoords` (`src/terms/smooth.rs:1774-1891`) is exactly the data structure pattern: a flattened `Array1<f64>` of "extra hyper-coordinates" whose layout is given by a `Vec<usize> dims_per_term`. The downstream machinery — `HyperDesignDerivative::from_implicit` (`src/solver/reml/mod.rs:2706-2721`), `DirectionalHyperParam::new_compact` (`src/solver/reml/mod.rs:3277`), the `is_penalty_like = false` flag set via `.not_penalty_like()` (`src/solver/reml/mod.rs:3314`; spatial call at `src/terms/smooth.rs:13105`), and the final entry into `RemlState::evaluate_unified_with_psi_ext` (`src/solver/reml/runtime.rs:8122`) — is already designed to accept *non-penalty-like, design-moving* ext-coordinates. Kernel-shape `ψ` and latent coordinates `t` are the same kind of object as far as the IFT plumbing is concerned. Both move the design `X`, both leave `S(ρ)` untouched (in the bare case), and both ship first- and second-derivative blocks through `HyperDesignDerivative`.

**Proposed Rust shape.** A sibling of `SpatialLogKappaCoords`:

```rust
#[derive(Debug, Clone)]
pub(crate) struct LatentCoordValues {
    /// Flattened latent vector: n_obs * latent_dim entries, row-major.
    values: Array1<f64>,
    latent_dim: usize,
    n_obs: usize,
    /// Which basis evaluates against t. Borrowed from the same basis registry
    /// that handles observed covariates.
    basis: Arc<dyn BasisOutput>,
}
```

with a `LatentCoordValues::build_x_ext_derivatives(&self) -> Vec<HyperDesignDerivative>` that produces one `HyperDesignDerivative` per latent column. For Duchon, Matern, sphere, periodic, and tensor bases, the landed `LatentCoordDesignDerivative` (`src/terms/basis.rs:4118`) routes through `LatentCoordValues::design_gradient_wrt_t_dispatch` (`src/terms/latent_coord.rs:825`): radial bases reuse the same radial derivative as `ψ`, while non-radial bases provide analytic jets. The remaining work is integration breadth, not new IFT math.

**Proposed Python shape.** Add a smooth-spec subclass that flags "the input to this smooth is a free latent vector, not a column of the data table":

```python
@dataclass
class LatentCoord:
    dim: int                       # latent dimensionality
    basis: Smooth                  # any existing Smooth describing the decoder
    init: Literal["pca", "random", "user"] = "pca"
    init_values: Any | None = None # if init="user"
```

`fit(..., latents={"t": LatentCoord(n=N, d=2, init="pca", aux_prior={...})}, smooths=[Duchon(centers=..., m=2, name="decoder")])` should construct the `LatentCoordValues` block, wire it into the same `DirectionalHyperParam` pipeline used for anisotropic `ψ`, and let REML run unchanged. Formula-side syntax can then bind `s(t, ...)` to that latent block.

**What's load-bearing.** The `is_penalty_like = false` derivation in `DirectionalHyperParam::new_compact` (`src/solver/reml/mod.rs:3277-3284`), backed by `HyperDesignDerivative::any_nonzero` (`src/solver/reml/mod.rs:2780-2786`), and downstream `validate_tk_ext_coords` (`src/solver/reml/runtime.rs:2521`) is what makes an ext-coordinate "design-moving" rather than "penalty-scaling". `LatentCoordValues` is design-moving. Roughly 70–90% of the code path is shared with anisotropic `ψ`; the new code is the per-observation indexing and the basis-side derivatives `∂X/∂t_n`, both of which are radial-kernel chain-rule applications of existing primitives.

### 4(b). Isometry-to-reference penalty

**Problem.** Bare `LatentCoordValues` has a gauge symmetry: any smooth invertible reparameterization `t → φ(t)` together with a re-fit decoder gives the same likelihood. Empirically this shows up as flat valleys in the inner objective and Procrustes alignment failures across seeds (this was the observation in `auto_71`). The IFT requires unique inner minima; flat valleys make the implicit Hessian rank-deficient and the implicit gradient ill-defined.

**Math.** Let `T: M → ℝ^p` be the decoder, with induced pullback metric `g_ij(t) = ⟨∂T/∂t_i, ∂T/∂t_j⟩`. Let `g^ref_ij(t)` be a reference Riemannian metric on the latent manifold (e.g. the identity, or the metric of a held-out chart). The isometry penalty is

```
P_iso(β, t; ρ_iso) = e^{ρ_iso} · ∫_M ‖g(t) − g^ref(t)‖_F^2 dt
```

approximated as a sum over either observation locations or a quadrature grid. `e^{ρ_iso}` is a standard penalty scaling and is selected by REML alongside the smoothness penalties.

**Interface.** The current analytic-penalty layer carries `IsometryPenalty` (`src/terms/analytic_penalties.rs:353`) alongside the older `BlockwisePenalty` machinery (`src/terms/smooth.rs:707`). The penalty Gram block is a function of `(β, t)` and therefore contributes a non-trivial `∂S/∂t` through `PenaltyDerivativeComponent` (`src/solver/reml/mod.rs:3188-3190`) and `HyperPenaltyDerivative` (`src/solver/reml/mod.rs:3027`) — the same derivative-carrier layer used by other hyper-penalty derivatives.

**When to use.** Whenever `LatentCoordValues` is in play and there is no auxiliary variable to break the gauge (§4(c)). Default to identity reference and let REML choose the strength.

### 4(c). Auxiliary-conditional-prior penalty

**Problem and theory.** Khemakhem et al. (2020, iVAE) show that if the latent `t` has a prior conditioned on an observed auxiliary `u` — `p(t | u)` from an exponential family with `u`-dependent natural parameters — then under mild assumptions the decoder is identifiable up to a permutation and component-wise transformation. This is the cleanest known structural identifiability result for nonlinear latent-variable models.

**Recasting in gamfit terms.** A conditional prior `p(t | u; η(u))` is, by Bayes, a penalty on the latent block whose Gram matrix is a function of `u`. Concretely, for a Gaussian conditional prior with `u`-dependent precision `Λ(u)`, the penalty contribution is `½ t_n^T Λ(u_n) t_n`, summed over `n`. This is a *blockwise diagonal penalty whose block is data-dependent*. The existing `BlockwisePenalty` infrastructure is the right home, with a new constructor `BlockwisePenalty::auxiliary_conditional_gaussian(u, precision_fn)`.

**Mechanism sparsity.** iVAE-style identifiability additionally benefits from sparsity of the Jacobian `∂T/∂t` (each latent affects only a few outputs). This is an L¹-on-a-block penalty over rows of the decoder coefficient matrix — exactly the kind of structured penalty the existing `OperatorPenaltySpec` machinery (`src/terms/basis.rs:2858`) handles when paired with the active-set inner solver (`src/solver/active_set.rs`).

**Interface.** One new smooth-spec field on `Latent`: `aux: Any | None = None` (column name or array), plus a `prior: Literal["gaussian", "gaussian-mixture", ...]` choice. Internals route to the new `BlockwisePenalty` constructor.

**Regularity conditions (math audit).** The claim that the aux-conditional prior `μ ‖t − h(u)‖²` breaks gauge *and* that `μ` is REML-selectable holds only under three explicit conditions, which the implementation must enforce:

1. **Normaliser present.** The marginal likelihood for `μ` must include the `(N/2) log μ` normaliser (or the appropriate equivalent for the chosen prior family). Without it, optimising over `μ` may pick `0`, `∞`, or be indifferent.
2. **Regularity of `h`.** `h` must be at least `C¹` (typically `C²`) for the IFT and Hessian-based gradient claims to hold.
3. **PD on the anchored subspace.** The conditional precision must be positive-definite on the subspace of `T` that `h(u)` actually constrains across the realised rows. (The orthogonal complement is where ARD or another prior must step in if needed.)

Without (1)-(3), the iVAE-style identifiability argument does not transfer; the prior may still regularise but it no longer provably breaks the gauge nor admits clean REML selection of `μ`.

### 4(d). ARD over latent dimensions

This is a composition, not a primitive. Given `LatentCoordValues` of dimension `d_max`, place one independent smoothness penalty per latent axis with its own `ρ_k`. With the proper marginal-likelihood normalisers — the `(N/2) log α_k` per-axis terms and the determinant/rank corrections — and *paired with a gauge fix from another source* (the aux-conditional prior of §4(c) or the isometry penalty of §4(b)), REML drives unused `ρ_k → +∞` (effectively zero contribution) on axes whose evidence does not justify them. The user reads off the intrinsic dimension as the count of finite `ρ_k`.

**Audit caveat — ARD alone is gauge-invariant.** The penalty `α_k ‖t_{·,k}‖²` is rotation-symmetric on T: a rotation of the latent axes simply re-shuffles which dim is "used" vs "unused" while leaving the penalty value unchanged. Without a paired gauge fix and without the normalisers above, ARD does not by itself break the gauge and does not by itself identify intrinsic dim — it only relaxes a basis you have already pinned. The composition (ARD + gauge fix + REML normalisers) is what does the work; the marketing line "ARD discovers intrinsic dim" is shorthand for that composition, not a property of ARD in isolation.

The sloppy-model "hyper-ribbon" diagnostic (Transtrum, Machta, Sethna) — the eigenspectrum of the REML Hessian at convergence — is a geometry-native cross-check: a clean gap between stiff and sloppy directions is the same statement as a clean gap between active and ARD-pruned latent axes, *once gauge is fixed*.

No new code. Recipe-level recommendation: `manifold_fit(..., latent_dim=d_max, dim_selection=True, aux_prior=...)` (or `isometry=True`) translates to one `LatentCoord` block named `t` with per-axis ARD plus the requested gauge-breaking prior.

### 4(e). Geodesic-acceleration Newton patch

**Problem.** On problems with strongly nonlinear residuals — and `LatentCoordValues` is one — the standard Gauss-Newton / PIRLS step systematically over-shoots along curved valleys. Transtrum's geodesic-acceleration correction adds a second-order term `δ_2 = −½ H^{-1} A(δ_1, δ_1)`, where `A` is the directional second derivative of the residual along the first-order step `δ_1`. Empirically this can reduce inner iterations substantially on curved latent fits. (auto_71 used the LM variant of this trick and observed exactly this behavior.)

**Implemented site.** The dense unconstrained path now factors through `solve_newton_direction_dense_with_factor` (`src/solver/pirls.rs:5040-5044`), then the geodesic block reuses that dense factor to solve the second RHS (`src/solver/pirls.rs:5086-5153`). It computes a central-difference estimate of the directional second derivative of the gradient along `δ_1`, solves a second system with the *same* factorization, and accepts the correction only when the Transtrum-Sethna norm criterion holds. Sparse, constrained, and arrow-Schur paths intentionally skip this correction.

**Gating.** Default off; opt in via `WorkingModelPirlsOptions::geodesic_acceleration: bool` (`src/solver/pirls.rs:1289`). Cost: two extra `WorkingModel::update` calls per accepted dense step for the finite-difference evaluations; no effect on paths where the gate or eligibility checks are false.

### 4(f). Riemannian latent updates for topology-matched `LatentCoord`

**Problem.** Basis topology alone is not enough for a latent coordinate. A cyclic basis can make `Φ(t)` continuous across the seam, but if the per-row coordinate update is Euclidean then an angular latent still has a spurious discontinuity at `2π → 0`. mGPLVM avoids this by putting the latent itself on the manifold (circle, sphere, torus, product manifolds) and optimizing with Riemannian updates.

**Implementation.** `LatentCoordValues` carries a `LatentManifold`:

```
Euclidean | Circle | Sphere { dim } | Interval { lo, hi } | Product(...)
```

Each variant supplies three closed-form operations from Absil/Mahony/Sepulchre / Manopt / Pymanopt: a retraction `R_t(ξ)`, tangent projection `P_t(v)`, and Euclidean-to-Riemannian Hessian action `H_R(t)[ξ]`. Circle is a wrapped scalar angle; sphere uses the normalization retraction `(t + ξ) / ||t + ξ||`, tangent projection `v - (v·t)t`, and the embedded-sphere Hessian conversion `P_t(H_E ξ) - <∇f_E,t> ξ` (equivalently, differentiating the projected gradient with the normal curvature correction).

**Composition with arrow-Schur.** The BA-grade arrow system stays arrow-shaped. Before point elimination, each row block is converted in place:

1. `g_t^(i)` is projected to `T_{t_i}M`.
2. `H_tt^(i)` is converted to the Riemannian Hessian action.
3. Each `H_tβ^(i)` column is tangent-projected.
4. The same Schur elimination / Sqrt-BA / Steihaug-CG machinery solves for a tangent `ξ_i`.
5. The accepted latent update is `t_i ← R_{t_i}(ξ_i)`, not `t_i += ξ_i`.

For the shipped manifolds the Riemannian metric is the ambient Euclidean pullback, so Steihaug-CG's dot products already compute the right trust-region norm once row vectors are tangent-projected. Embedded constrained factors pin the normal block only to keep the existing ambient Cholesky path nonsingular; RHS and cross terms remain tangent, so the solved row step is a tangent vector.

**Auto-dispatch.** Python `LatentCoord(..., manifold="auto")` infers the natural manifold from the consuming basis: periodic/cyclic 1-D bases become `Circle`, spherical bases become `Sphere { dim }`, tensor bases with periodic margins become a product (torus/cylinder), and ordinary Duchon/Matérn/TPS bases stay `Euclidean`. Explicit `manifold="circle"`, `"sphere"`, `"torus"`, `"cylinder"`, or interval objects override inference.

**Gauge framing.** This is not a numerical convenience. Restricting the update to `S¹`, `S²`, or a product manifold is itself a gauge restriction: the latent is identifiable up to the manifold's global isometries (for example a single rotation per cycle) rather than an arbitrary diffeomorphism of Euclidean coordinates.

## 5. Worked examples

Each example is a 3–5 line specification that replaces a foreign library. APIs follow the existing `gamfit.fit` conventions and the new `LatentCoord` configuration object.

### GP-LVM (replaces GPy.GPLVM)

```python
model = gamfit.fit(
    response=Y,                            # (n, p) observation matrix
    smooths=[Duchon(centers=..., m=2, length_scale=None, name="decoder")],
    latents={"t": LatentCoord(n=N, d=2, init="pca", aux_prior={...})},
    family="gaussian",
)
t_hat = model.latent("t")
```

REML selects the decoder smoothness; `LatentCoord.init="pca"` gives the standard PCA warm start; the gauge symmetry is broken by adding `IsometryPenalty` or by supplying an auxiliary prior. `dim_selection=True` *given* one of those gauge fixes (plus the proper marginal-likelihood normalisers) auto-discovers the intrinsic dimension; without a paired gauge fix, ARD on its own is rotation-symmetric and does not identify intrinsic dim (§4(d) audit caveat).

### iVAE-style identifiable representation (replaces hand-rolled VAE)

```python
model = gamfit.fit(
    response=X,                            # (n, p) observed
    smooths=[Duchon(centers=..., m=2, name="decoder")],
    latents={"t": LatentCoord(
        n=N, d=8,
        aux_prior={"u": environment_id, "family": "ridge", "strength": "auto"},
        dim_selection=True,
    )},
    penalties=[SparsityPenalty(target="decoder", strength="auto")],
    family="gaussian",
)
```

The `aux_prior` argument routes to the new `BlockwisePenalty::auxiliary_conditional_gaussian` constructor; `SparsityPenalty` adds the L¹-on-rows penalty via the existing active-set path. Identifiability is the Khemakhem theorem under the normaliser / regularity / positive-definiteness conditions in §4(c); the selector for *how strongly* to enforce it is REML over the per-block ρ.

### Fisher-Rao color manifold (the cogito work)

```python
class FisherWorkingModel(WorkingModel):
    def update(self, beta):
        # standard mean/variance/eta computation, but W = output Fisher info
        ...

model = gamfit.fit(
    response=color_responses,
    smooths=[Sphere(...), Duchon(centers=stimuli, m=2)],
    latents={"t": LatentCoord(n=N, d=2, aux_prior={...})},
    working_model=FisherWorkingModel(),
)
```

The Fisher inner product is *just* the choice of `W` inside `WorkingModel::update`. Nothing about REML, IFT, or basis evaluation changes — the standard machinery already differentiates through arbitrary `W` because PIRLS is written for it (`src/solver/pirls.rs:526-529`).

### Principal-curve fitting (Hastie–Stuetzle)

```python
model = gamfit.fit(
    response=points,                       # (n, d)
    smooths=[BSpline(knots=...)],
    latents={"t": LatentCoord(n=N, d=1, init="pca", aux_prior={...})},
    family="gaussian",
)
curve = model.predict_along("t", np.linspace(0, 1, 200))
```

Principal curves are just `LatentCoord(dim=1)` / `LatentCoordValues` with a smooth decoder. The smoothing parameter is selected by REML rather than by cross-validation. Hastie and Stuetzle's iterative projection is what the inner solver does anyway.

### Topological coordinate after TDA

```python
# 1. Persistent cohomology says: one nontrivial H^1 class, so latent is S¹.
# 2. Topology is the basis choice:
model = gamfit.fit(
    response=Y,
    smooths=[PeriodicSplineCurve(period=2*np.pi, ...)],
    latents={"t": LatentCoord(n=N, d=1, aux_prior={...})},
)
```

This is the cleanest expression of the framing. TDA tells you *which* basis; gamfit gives you the *fit* under that topology, with REML choosing the regularity strength and the existing IFT pipeline propagating gradients.

## 6. Mechanical mapping to existing code

This section is for the maintainers; it cites the precise sites that new work would touch.

**The ext-coordinate machinery.** The end-to-end pipeline for design-moving hyperparameters is:

1. `SpatialLogKappaCoords::from_length_scales_aniso` (`src/terms/smooth.rs:1844-1891`) constructs the flat ψ vector with a `dims_per_term` layout.
2. `ImplicitDesignPsiDerivative` (`src/terms/basis.rs:4054`) provides `∂X/∂ψ` without materializing the dense matrix; `LatentCoordDesignDerivative` (`src/terms/basis.rs:4118`) does the analogous `∂X/∂t` work for latent coordinates.
3. `HyperDesignDerivative::from_implicit` (`src/solver/reml/mod.rs:2706-2721`) wraps the implicit operator into the storage type the REML machinery expects.
4. `DirectionalHyperParam::new_compact(...)` + `.not_penalty_like()` (`src/solver/reml/mod.rs:3277`, `src/solver/reml/mod.rs:3314`) packages the per-direction first/second derivatives and tags the coordinate as design-moving.
5. `RemlState::build_tau_unified_objects_from_bundle` call site (`src/solver/reml/runtime.rs:8137-8143`) builds the `HyperCoord` list, ext-pair functions, and fixed drift derivatives.
6. `RemlState::evaluate_unified_with_psi_ext` (`src/solver/reml/runtime.rs:8122`) evaluates the unified REML/LAML score and its gradient with the ψ/ext-coords folded in.

A `LatentCoordValues` block uses every step of this pipeline. Step 1 becomes per-observation rather than per-term, but the storage shape is the same. Step 2 is the same operator with a different chain rule. Steps 3–6 are unchanged. The `is_penalty_like = false` semantics already isolate the design-moving case correctly.

**The inner Newton solver.** `src/solver/pirls.rs:4565-5165` is the LM-damped Newton outer loop. The Newton direction is computed at one of several sites depending on backend: sparse-native factor/solve (`src/solver/pirls.rs:4930-4949`), `solve_newton_directionwith_linear_constraints` (`src/solver/pirls.rs:4962`), `solve_newton_directionwith_lower_bounds` (`src/solver/pirls.rs:4971`), arrow-Schur (`src/solver/pirls.rs:4979-5038`), and dense-factor solve (`src/solver/pirls.rs:5040-5044`). Geodesic acceleration is a dense-path post-processing step (`src/solver/pirls.rs:5086-5153`): with the factorization still in scope, estimate the gradient's second directional derivative along `δ_1`, then re-solve the same system for `δ_2`. The path is gated by `WorkingModelPirlsOptions::geodesic_acceleration`.

**REML outer loop.** `src/solver/outer_strategy.rs::OuterProblem` (struct at `src/solver/outer_strategy.rs:4307`, workflow construction at `src/solver/workflow.rs:1968-1974`) drives the BFGS / compass-search loop in ρ. Adding ext-coords (`ψ`, latent `t`) lengthens the optimization vector but does not change the loop structure; this is already exercised for anisotropic `ψ`.

**Penalty blocks.** `BlockwisePenalty` (`src/terms/smooth.rs:707`) and `KroneckerPenaltySystem` (`src/terms/smooth.rs:855`) already accept arbitrary user-constructed Gram blocks. The analytic penalty layer now carries isometry, sparsity, and ARD penalties (`src/terms/analytic_penalties.rs:353`, `src/terms/analytic_penalties.rs:1141`, `src/terms/analytic_penalties.rs:1844`), with `PenaltyDerivativeComponent` (`src/solver/reml/mod.rs:3188`) carrying the `∂S/∂t` and `∂S/∂ρ` blocks.

**What is new vs. extended.** Landed modules now include `src/terms/latent_coord.rs` (`LatentCoordValues` at `src/terms/latent_coord.rs:561`) and `src/terms/analytic_penalties.rs` (`IsometryPenalty` at `src/terms/analytic_penalties.rs:353`). The derivative extension lives beside the basis machinery as `LatentCoordDesignDerivative` (`src/terms/basis.rs:4118`), rather than as a `BasisOutput` trait method. No `WorkingModel` change is required for the Fisher-Rao use case; the trait is already general enough.

## 7. Open questions and risks

**The gauge symmetry is mathematically load-bearing.** A bare `LatentCoordValues` block, with no auxiliary variable and no isometry penalty, has a continuous family of equivalent inner minima. The IFT requires *unique* (locally isolated) minima with full-rank inner Hessian; flat valleys violate this. Empirically this was seen in the `auto_71` runs: geodesic-LM converged to identical R² across seeds with Procrustes ≈ 0.99, i.e. the fit was great but the latent was indistinguishable from a smooth diffeomorphic image of any other. The proposal is to make `LatentCoord` *refuse to fit* unless at least one of {`aux_prior`, `IsometryPenalty`} is supplied. (ARD on its own is *not* in this list — per the §4(d) audit caveat, `α_k ‖t_{·,k}‖²` is rotation-symmetric and does not break gauge without a paired prior. ARD is a useful *companion* to a gauge fix, not a substitute.) The error message should explain why. This is a stronger position than mgcv takes about its own corner cases and is appropriate here because the failure mode is silent.

**Composability under multiple new primitives.** The REML/IFT machinery is currently exercised with one `ψ` block and many `ρ` blocks. With `LatentCoordValues` (per-observation ext-coords), an auxiliary-conditional penalty (data-dependent Gram), and isometry-to-reference (β-and-t-dependent Gram) all in one fit, the unified Hessian gains substantial off-diagonal coupling. The Lean scaffolding (`src/solver/reml.lean`) proves the three-term LAML decomposition without assuming block-diagonality, so the math is fine, but the *numerical* conditioning of the joint Hessian and the *runtime* cost of the IFT solve in this regime need empirical study. A small scaling study (1, 2, 3, 4 simultaneously-active new primitive types) before declaring the manifold workflow stable would be appropriate.

**Arrow / Schur cost — precise statement (math audit).** Earlier drafts said "per-row latent `t_i` has block-diagonal Hessian; arrow/Schur eliminates the latent cheaply." That is true at the cost level but the explanation must be precise. After Schur elimination, `log|H| = log|H_tt| + log|Schur|`. The first term factorises trivially because `H_tt` is block-diagonal across rows. For the second, `∂/∂t_i log|Schur| = tr(Schur⁻¹ · ∂Schur/∂t_i)` where `Schur⁻¹` is dense in all `t`; but `∂Schur/∂t_i` is a rank-≤d update because only row `i` of `Φ` moves with `t_i`. Per outer iteration the cost is therefore one dense `Schur⁻¹` formation (shared across rows) plus N cheap per-row traces — still `O(N + decoder-cost)`, not `O(N²)`. The arrow shape holds at the cost level; the gradient formula carries a shared factor that is a one-time setup per outer iteration.

**REML behavior near-singular inner Hessian.** When `Latent` is used without gauge-fixing, the inner Hessian is rank-deficient by construction in the gauge direction, and the standard ridge regularization in `solve_newton_direction_dense` will hide this rather than reveal it. The diagnostics for "this fit is in a flat valley" need to be surfaced — most cleanly as a check on the smallest eigenvalue of the *unregularized* Hessian, gated by a flag. This is also useful for diagnosing user-data pathology in the existing GAM workflow, so the investment is not single-use.

**The Python `manifold_fit` ergonomic.** Composing `Latent`, `IsometryToReference`, ARD, and a Fisher working model in one `fit()` call is verbose. A thin `gamfit.manifold_fit(...)` wrapper that constructs the right combination from a handful of high-level knobs (`latent_dim`, `topology`, `auxiliary`, `metric`) is justified once §4(a)-(d) are implemented and the defaults are settled. This is a documentation and ergonomics task, not an architectural one.

**Maintainers' clarifications wanted.**
- The per-observation `∂X/∂t_n` method now lives in `LatentCoordDesignDerivative` (`src/terms/basis.rs:4118`) plus `LatentCoordValues::design_gradient_wrt_t_dispatch` (`src/terms/latent_coord.rs:825`), not on `BasisOutput` (`src/terms/basis.rs:270`). Is that the intended long-term ownership boundary?
- Does `KroneckerPenaltySystem` support penalty blocks whose Gram is *not* a Kronecker product but depends on data (auxiliary-conditional precision)? If not, the auxiliary-conditional penalty needs to route through `BlockwisePenalty` only.
- The `outer_ift_residual_energy_cache` (`src/solver/reml/runtime.rs:54`) keys on hashes of ρ. With ext-coords (`ψ`, `t`) in scope, the cache key should include them; is there an existing convention for extending the key in the anisotropic `ψ` path that `LatentCoordValues` should follow?
- The LM warm-start fields (`initial_lm_lambda` at `src/solver/pirls.rs:1274`, `final_lm_lambda` at `src/solver/pirls.rs:1414`) suggest an existing inner-solver warm-start strategy. For `LatentCoordValues` whose values change between REML outer iterations, is warm-starting `t` from the previous outer iterate the intended behavior, and is there a hook for that?

A short reply to these from the maintainers would unblock a prototype `LatentCoord` patch that demonstrates the GP-LVM and principal-curve examples end-to-end on the existing test data. The remaining items (isometry, auxiliary prior, geodesic acceleration) are independent and can land in any order.

## 8. Audit revisions

This document was revised in response to a math-audit pass on the original optimistic draft. Tightened claims:

- **§2 tier-assignment table (Manifold/GP-LVM row, intrinsic-dim row).** The intrinsic-dim row now states that ARD discovers intrinsic dim *given* a paired gauge fix from (a)/(b)/(c) and the proper marginal-likelihood normalisers — not as a standalone consequence of ARD.
- **§4(c) aux-conditional prior.** Three explicit regularity conditions added (normaliser present, `h` at least `C¹`, conditional precision PD on the anchored subspace). The earlier draft asserted "`μ` is REML-selectable" without these.
- **§4(d) ARD over latent dimensions.** Rewritten to state that `α_k ‖t_{·,k}‖²` is rotation-symmetric and therefore does *not* break gauge by itself. The composition (ARD + paired gauge fix + REML normalisers) is what does the work. The earlier draft conflated "ARD" with "gauge fix + intrinsic-dim discovery."
- **§5 worked GP-LVM example.** Removed the implication that `dim_selection=True` / ARD alone is a sufficient gauge fix.
- **§7 open questions (Latent refuse-to-fit policy).** ARD removed from the list of acceptable gauge-breaking choices, with the audit caveat cited.
- **§7 composability under multiple new primitives.** Promoted the earlier footnote about Schur/arrow cost into the main complexity claim: cost is arrow-shaped, but the REML `log|H|` gradient carries a shared `Schur⁻¹` factor handled as one-time-per-outer-iteration setup plus N rank-≤d per-row traces.

Source of caveats: math-audit findings summarised in `/Users/user/.claude/projects/-Users-user-Manifold-SAE/memory/project_gamfit_composition_engine.md` (the "Math-audit findings" section).
