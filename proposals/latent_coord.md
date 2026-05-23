# Proposal: `LatentCoord` — per-row latent coordinates as a first-class gamfit parameter

Status: draft. Author: scottsauersc@gmail.com.  
Companion prototype: `/Users/user/Manifold-SAE/experiments/auto_75.py`.

## 1. Motivation

gamfit today fits

```
y = Φ(x) · β + ε,   penalty(β, ψ),   x observed
```

where `β` are smoothing-spline coefficients, `ψ` are kernel / anisotropy
hyper-parameters (currently flowing through `SpatialLogKappaCoords`), and `x`
is an observed covariate. The REML/LAML loop selects smoothing; an
IFT-warm-started nested optimizer moves `ψ` because `∂Φ/∂ψ` is computable
from the radial basis.

The proposal: promote some covariates from observed to **latent**:

```
y = Φ(t) · β + ε,   penalty(β, ψ),   t ∈ ℝ^(N×d) estimated per-row.
```

Mechanically this is the same problem as `ψ` already solves: we need
`∂Φ/∂t` per row, and that's structurally the same radial-basis-gradient
that gamfit's spatial machinery already computes — except differentiating
with respect to the *first* argument of the kernel rather than the
log-anisotropy. Everything downstream (IFT-warm-started Newton on the
inner system, REML for the outer ridges, persistent warm-start) is reused
verbatim.

Use case: **GP-LVM and principal-manifold inference** (Lawrence 2005,
Titsias 2010), entirely in gamfit primitives. This subsumes a non-trivial
set of workloads — manifold learning, latent-trait inference, identifiable
nonlinear ICA via iVAE-style auxiliary priors — that currently require
GPy / sklearn `IsoMap` or hand-rolled alternation (see this repo's
`experiments/auto_71.py` for a geodesic-LM gauge-unfixed example, and
`auto_74.py` for the alternation-then-ICA workaround).

We already ship `gaussian_reml_fit_positions{,_backward}` for the 1-D
case (B-spline positions with `grad_t` exposed). `LatentCoord` is the
*N-D Duchon* generalization of that same idea, lifted to a configuration
type rather than a bespoke function.

## 2. Math

### 2.1 Loss

For Gaussian-identity, profiling out σ²:

```
ℒ(t, β, ψ, λ) = ½‖y − Φ(t; ψ) β‖² + ½λ βᵀ S(ψ) β
                + R_id(t, u)                              # identifiability
                − ½ log|Σ(λ, ψ, t)|                       # REML log-det
```

`Φ(t; ψ)_{i,k} = K_ψ(t_i, c_k)` is the design row built from the latent
coordinate `t_i ∈ ℝ^d` and the basis centers `c_k`. For Duchon thin-plate
splines, `K_ψ` is radial: `K(r)` with `r = ‖t_i − c_k‖`, modulated by
ψ-anisotropy exactly as today.

### 2.2 Gradients

Let `g_i = ∂Φ_i / ∂t_i ∈ ℝ^(K × d)`. For a Duchon kernel
`Φ_{i,k} = φ(r_{ik})` we have `∂Φ_{i,k}/∂t_i = φ'(r_{ik}) · (t_i − c_k)/r_{ik}`,
the same radial-gradient routine that
`SpatialLogKappaCoords::apply_tospec` invokes for `∂Φ/∂ψ` (anisotropy
just rescales components of `(t_i − c_k)` before contraction).

The gradient w.r.t. `t_i` of the residual is row-local:

```
∂(½‖y − Φβ‖²) / ∂t_i = −(y_i − Φ_i β) · (g_i β)        # shape (d,)
```

so the per-row Hessian block (Gauss–Newton) is `(g_i β)(g_i β)ᵀ`. The
joint Hessian `H_tt` is block-diagonal in rows (huge — `N·d × N·d` — but
sparse-by-blocks), `H_tβ = − Φ_i ⊗ (g_i β) + …` is dense in β but only
row-local in t. This is the *same* structure gamfit's matrix-free CG
already exploits along the ψ direction (`hyper_dirs`).

**Arrow shape — careful statement (per math audit).** The cost is
arrow-shaped, but the REML `log|H|` Occam-term gradient is not literally
`Σ_i f(t_i)`. After Schur elimination,
`log|H| = log|H_tt| + log|Schur|`. The first term factorises trivially
because `H_tt` is block-diagonal in rows. For the second,
`∂/∂t_i log|Schur| = tr(Schur⁻¹ · ∂Schur/∂t_i)`, with `Schur⁻¹` dense in
all `t`. However, only row `i` of `Φ` moves with `t_i`, so
`∂Schur/∂t_i` is a rank-≤d update. Per outer iteration the cost is one
dense `Schur⁻¹` formation (shared across all rows) + N cheap per-row
traces — `O(N + decoder-cost)`, not `O(N²)` and not `O((Nd)³)`. The
arrow shape holds at the cost level; the gradient formula carries a
shared factor that is a one-time setup per outer iteration.

### 2.3 Gauge symmetry — load-bearing caveat

The bare data-fit loss `½‖y − Φ(t) β‖²` is invariant under any
diffeomorphism `t ↦ φ(t)`, because we can absorb `φ` into a re-fit of
`β` against the relabeled basis. Concretely:

- Translation / rotation of `t` ↔ relabeling of centers.
- Smooth coordinate change ↔ basis transformation; β re-fits.

The penalty `βᵀ S β` *does* depend on coordinates (it penalizes
spline-curvature in `t`-space), but is invariant under isometries
(rotation + translation), which still leaves at least a `d(d+1)/2`-dim
gauge orbit per fit. **IFT is singular along this orbit** — the inner
Hessian has zero eigenvalues, so the nested optimizer's predictor step
blows up. `auto_71.py` in the companion repo demonstrates this: geodesic
LM hits R² = 0.640 but Procrustes disparity = 0.99 across seeds —
solutions wander freely along the gauge.

Three model-level fixes, all exposed as `LatentCoord` configurations:

**(a) `aux_prior`** — conditional prior `p(t | u)` with `u` observed
(iVAE; Hyvärinen 2019, Khemakhem et al. 2020). Add a regulariser
`R_id(t, u) = ½ τ · ‖t − h(u)‖²` where `h` is a small learned map
(ridge / shallow GAM). Different `φ` (latent reparameterisations) give
different `‖t − h(u)‖²`, so the orbit collapses to a single
representative. This is the **principled fix**; identifiability of the
resulting `t` follows from the iVAE result (assuming `u` varies enough
across rows).

**Regularity conditions (per math audit) — these must hold for the
claim that `τ` is REML-selectable and gauge is broken:**

1. The marginal likelihood for `τ` must include the `(N/2) log τ`
   normaliser (or the equivalent for the chosen prior family).
   Without it, optimising over `τ` can degenerate to `0`, `∞`, or be
   indifferent — the data-fit/penalty trade-off lacks the term that
   makes the trade-off well-posed.
2. `h` must be at least `C¹` (typically `C²` in practice) so the IFT
   and Hessian claims downstream are well-defined.
3. The conditional precision `τ · I` (or `Λ(u)` in the Gaussian
   conditional case) must be positive-definite on the anchored
   subspace of `T` — i.e. the subspace `h(u)` actually constrains
   across the realised rows.

If any of (1)-(3) fails, the gauge-breaking and `τ`-selection claims
do not go through. The implementation should enforce all three (the
normaliser as a code path, `h` regularity by the basis choice, PD by
a runtime check on the row-wise design).

**(b) `orthogonality`** — penalize `‖TᵀT/N − I_d‖²_F` after centering
`T ← T − T̄`. Cheap to compute, kills the rotation + scale gauge, but
doesn't help with the diffeomorphism gauge for `d > 1`. Useful as a
warm-start regularizer or a cheap second-best.

**(c) `dim_selection`** — ARD per latent axis, with REML selecting the
per-axis precisions `α_j`. **Not a standalone gauge fix.** The penalty
`α_j ‖t_{·,j}‖²` is rotation-symmetric on `T`: a rotation can re-shuffle
which axis is "used" vs "unused" without changing the penalty value.
ARD discovers intrinsic dim only *given* a paired gauge fix from (a) or
(b) and only when the marginal likelihood includes the proper
normalisers — the `(N/2) log α_j` per-axis terms and the
determinant/rank corrections. Under those conditions, REML drives
`α_j → ∞` on axes the data does not support and the user reads off
intrinsic dim as the count of finite `α_j`. The intended use is
**combined**: aux-prior (a) breaks the rotation gauge, ARD (c)
identifies which axes carry signal. ARD on its own does not break the
gauge and does not by itself identify intrinsic dim. (Source: math
audit on the original draft.)

## 3. API sketch

### 3.1 Python

```python
import gamfit
from gamfit import LatentCoord

# Replaces an observed-x covariate with a learned t of dimension d.
t = LatentCoord(
    n=N,
    d=4,
    init="pca",                 # 'pca' | 'random' | array (N, d)
    bounds=(-3.0, 3.0),         # per-axis clamp (broadcast or per-d)
    aux_prior=dict(             # (a) iVAE-style
        u=rgb,                  # (N, p) observed auxiliary
        family="ridge",         # 'ridge' | 'duchon'
        strength=1.0,           # τ in the math
    ),
    # OR
    orthogonality=0.1,          # (b)
    # OR
    dim_selection=True,         # (c)   ⇒ marginal-likelihood prunes d
)

model = gamfit.GAM(
    y ~ s(t, basis="duchon", m=2, nullspace_order="degree2")
       + s(observed_x, basis="duchon", m=2),
    data=df,
    latents={"t": t},
)
model.fit()                      # joint outer over (t, ψ, λ); REML inner
print(model.latents["t"].values) # (N, d) posterior mean
```

### 3.2 Rust

A new sibling of `SpatialLogKappaCoords`:

```rust
// src/terms/smooth.rs  (alongside SpatialLogKappaCoords)
pub(crate) struct LatentCoordValues {
    /// Flattened (N, d) latent matrix, row-major.
    values: Array2<f64>,
    /// Which design-term consumes this latent (one term per LatentCoord).
    term_index: usize,
    /// Identifiability mode.
    id_mode: LatentIdMode,
}

pub(crate) enum LatentIdMode {
    AuxPrior { u: Array2<f64>, g: AuxPredictor, tau: f64 },
    Orthogonality { weight: f64 },
    ReMlDimSelect,
}

impl LatentCoordValues {
    /// Mirrors SpatialLogKappaCoords::apply_tospec: re-materializes the
    /// design column block for the consuming term at the current t.
    pub(crate) fn apply_tospec(
        &self,
        spec: &TermCollectionSpec,
    ) -> Result<TermCollectionSpec, ...>;

    /// Mirrors the radial-gradient routine already used by ψ.
    /// Returns (N, K, d) jet for the consuming term.
    pub(crate) fn design_gradient_wrt_t(
        &self,
        spec: &TermCollectionSpec,
    ) -> Result<Array3<f64>, ...>;
}
```

The outer optimizer receives `t` as a new block of `hyper_dirs` (existing
machinery — `try_build_spatial_log_kappa_hyper_dirs` already builds these
for ψ; the new builder is `try_build_latent_coord_hyper_dirs`).

## 4. Mechanical reuse map

| Existing piece | Reused as | Notes |
| --- | --- | --- |
| `SpatialLogKappaCoords::apply_tospec` | template for `LatentCoordValues::apply_tospec` | both re-materialize Φ |
| `spatial_log_kappa_hyper_dirs_frominfo_list` | template for `latent_coord_hyper_dirs_frominfo_list` | both emit `DirectionalHyperParam` |
| `try_build_spatial_log_kappa_hyper_dirs` | template for `try_build_latent_coord_hyper_dirs` | dispatch at outer-fit assembly |
| `solver::estimate` IFT loop | unchanged | new hyper_dirs flow through |
| `solver::persistent_warm_start::last_ift_prediction_residual` | unchanged | same diagnostic |
| `solver::outer_strategy::EfsEval` | unchanged | same eval-and-gradient shape |
| `duchon_radial_jets` in `terms/basis.rs` | now also queried via `design_gradient_wrt_t` | same routine; first-arg derivative |
| `gaussian_reml_fit_positions_backward_impl` + `contract_position_gradient` | shipped 1D special case → generalize to ND | already returns `grad_t` |

**New pieces required:**
1. `LatentCoordValues` struct + the three id-mode variants.
2. A new term-spec attribute `latent: Option<LatentCoordRef>` on the
   smooth-term builder so a term consumes `t` instead of an observed col.
3. Outer-fit dispatch to build `hyper_dirs` for any registered
   `LatentCoord`s, identical pattern to kernel-shape `ψ`.
4. `dim_selection=True` path: when set, expand `d` by one and let
   REML decide; needs an outer loop over `d` analogous to the existing
   smoothing-grid search.

Crucially: **no new optimizer**, **no new IFT code**, **no new
matrix-free CG**. The existing ext-coordinate machinery covers it.

## 5. Test plan

Reproduce `auto_74`'s stacked CV R² = 0.608 on cogito-L40 color
centroids using *only* a `LatentCoord`-equipped GAM config — no
hand-rolled alternation, no FastICA, no per-fold ICA refit.

```
y = Z_top16  ∈ ℝ^(N × 16)
config:
  s(hue, periodic, m=2)               # observed
  + s(sat, val, m=2, ns=degree2)      # observed
  + s(t, m=2, ns=degree2,             # LATENT
       latent=LatentCoord(d=4, aux_prior=u=rgb))
```

Expected:
- CV R² ≥ 0.60 (matches auto_74 stacked).
- Procrustes disparity across seeds ≤ 0.2 (auto_71 was 0.99 unfixed).
- REML-selected `d` ∈ {3, 4} when `dim_selection=True` is enabled.

Empirical baseline observed in the prototype (`auto_75.py`, K_PC=16):
- observed-only (hue + sv) CV R² = 0.41 (the K_PC=16 ceiling, vs 0.32 at K_PC=64).
- LatentCoord d=4 with τ=0.5 aux-prior on RGB: CV R² ≈ 0.50 (early).
- The prototype's pure-Python out-of-fold `t_test` refinement is the
  bottleneck for full parity with auto_74; the Rust LatentCoord with
  IFT-warm-started `grad_t` would close this gap with better step
  control (and at production speed).

The companion prototype `experiments/auto_75.py` simulates this in
pure Python (using gamfit's existing primitives + a hand-rolled outer
on `t`) to validate the loss surface and identifiability gain *before*
the Rust work lands.

## 6. Open questions

1. **`d` initialization for `dim_selection`**: brute-force `d` sweep,
   or jet-style continuous relaxation (one column at a time, scaled by
   a regularizer that REML drives to zero)?
2. **Auxiliary-prior choice for `g_φ`**: ridge sufficient, or do we
   need a full nested GAM for the conditional mean?
3. **Periodic latent axes**: trivial to support (`periodic_per_axis`
   already exists) but exposes the question of *which* axes the user
   wants periodic. Default false; let user opt in.
4. **Backprop through latent**: should `LatentCoord` be backprop-visible
   to PyTorch (`gamfit.torch`) so the latent participates in upstream
   gradients? Probably yes — mirrors the existing `positions_backward`.
5. **Scaling**: for `N = 10⁶`, `(N, d)` is small but the per-row
   row-local Hessian needs the matrix-free CG path. Confirm this falls
   out of the existing infrastructure without new code.

## 7. Audit revisions

This document was revised in response to a math-audit pass on the
original optimistic draft. Tightened claims:

- **§2.2 arrow Hessian.** Added the explicit caveat that the REML
  `log|H|` gradient is *not* literally `Σ_i f(t_i)` — it carries a
  shared `Schur⁻¹` factor. Per outer iteration: one dense `Schur⁻¹`
  formation + N rank-≤d per-row traces. Cost remains
  `O(N + decoder-cost)`; arrow shape holds at the cost level. Earlier
  draft framed rows as "completely independent."
- **§2.3(a) aux_prior.** Added three explicit regularity conditions
  required for the gauge-breaking and `τ`-selection claims to hold:
  `(N/2) log τ` normaliser present, `h` at least `C¹`, conditional
  precision PD on the anchored subspace. Earlier draft asserted "the
  principled fix" without these.
- **§2.3(c) dim_selection.** Rewritten to state that ARD per latent
  axis is **not** a standalone gauge fix — `α_j ‖t_{·,j}‖²` is
  rotation-symmetric. ARD discovers intrinsic dim only given (a) or
  (b) and only with the proper REML normalisers. Earlier draft
  claimed REML on `t` alone "gives an automatic intrinsic-dim
  estimator."

Source: math-audit findings in
`/Users/user/.claude/projects/-Users-user-Manifold-SAE/memory/project_gamfit_composition_engine.md`.
