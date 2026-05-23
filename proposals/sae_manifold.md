# Proposal: SAE and SAE-manifold as gamfit configurations

Status: draft RFC
Audience: gamfit maintainers
Companion proposals: `composition_engine.md` (three-tier engine), `latent_coord.md` (per-row latent field)
Companion empirical work: `/Users/user/Manifold-SAE` (Curve-SAE benchmark)

## 1. Premise

Sparse autoencoders (SAEs) are routinely framed as "a separate
architecture for interpretability." This proposal argues the opposite:
**a linear SAE is a particular configuration of gamfit's three-tier
engine, and a non-linear "SAE-manifold" — the natural fix for
representation shattering — is the same configuration with one
additional primitive (`LatentCoord`) already proposed in
`latent_coord.md`.** There is no new optimizer, no new objective family,
no new identifiability theory required. There is a tier assignment and
a choice of priors.

The three-tier reading of gamfit is laid out in `composition_engine.md`:

- **β** — decoder / smooth coefficients, solved by the penalised-LS
  Newton inner solver (`src/solver/pirls.rs`).
- **ψ** — nonlinear design parameters, transported through the inner
  solution by the implicit function theorem (`HyperDesignDerivative`,
  `src/solver/reml/mod.rs:2653`).
- **ρ** — penalty / hyperparameter scalings, selected by the marginal
  likelihood in the REML outer loop (`src/solver/reml/runtime.rs`).

"Methodspace" is the cross-product of (tier assignment) × (choice of
basis / topology) × (choice of identifiability priors). The point of
this proposal is to show that SAE and SAE-manifold are two more rows
in that table; that the SAE row is already buildable with the
shipping primitives; that the SAE-manifold row needs only what
`latent_coord.md` already proposes; and that the well-known
"shattering" pathology of linear SAEs dissolves under the manifold row
because the REML Occam factor can prefer one curved atom over `K`
linear shards (under proper prior normalisation, controlled gauge, and
smooth-decoder effective dimension growing slower than the required
shard count — see §5.3 and the math-audit caveats).

## 2. The tier-assignment table

| Method | β | ψ | ρ | Identifiability priors |
|---|---|---|---|---|
| GAM | smooth coefficients | (none) | smoothness | Duchon `S(ρ)` penalty (`gamfit/smooth.py:79`) |
| Anisotropic-Matern fit | smooth coefficients | log length-scales (`SpatialLogKappaCoords`, `src/terms/smooth.rs:1765`) | smoothness + per-axis length | smoothness + Matern hyperprior |
| Manifold / GP-LVM | decoder coefficients | per-row latent `t_i` (`LatentCoord`, `latent_coord.md` §3) | smoothness + ARD + topology | isometry / ARD / smoothness / topology-via-basis |
| **Linear SAE** | dictionary `D` + codes `a_i` | (none) | sparsity strength `λ_sp` + dict size `K` | L¹-like on `a_i` (active-set, `src/solver/active_set.rs`); Occam on `K` via REML / `compare_models` |
| **SAE-manifold** | per-atom decoder coeffs `B_k` | soft assignment `a_i` + on-atom coord `t_ik` (`LatentCoord` × K atoms) | sparsity + per-atom dim + smoothness | sparsity on `a_i`; ARD per `t_ik` axis; smoothness on each `B_k` |

The middle four rows are already implementable: GAM ships today,
anisotropic-Matern fits today, GP-LVM lands with `LatentCoord` (a
single global latent block), linear SAE lands with the
already-shipped active-set + `compare_models` combination plus a
non-negativity / sparsity penalty on the code block. The bottom row
is the new contribution and is treated in §3.

The point is that "SAE" and "SAE-manifold" do not require a separate
code path. They are choices about which parameter block lives in which
tier and which prior is attached.

## 3. The SAE-manifold formulation

### 3.1 Variables

Observed: `Z_i ∈ ℝ^p`, `i = 1 … N` (e.g. residual-stream activations
at a layer).

Parameters, partitioned by tier:

- **β** = `{B_k}_{k=1..K}`, where `B_k ∈ ℝ^{M_k × p}` are the
  spline-coefficient blocks of atom `k`'s decoder. `M_k` is the
  basis size of atom `k`. β lives in the Newton inner solve.

- **ψ** decomposes into two per-observation sub-blocks:
  - `a_i ∈ ℝ^K_{≥0}` — soft assignment of observation `i` to each
    atom. Constrained non-negative; the sparsity prior on `a_i`
    encourages most entries to be exactly zero.
  - `t_ik ∈ ℝ^{d_k}` — on-atom coordinate of observation `i`
    *if* it is assigned to atom `k`. `d_k` is the intrinsic
    dimension of atom `k`. `t_ik` is materially present only when
    `a_ik > 0`; otherwise it is an unconstrained nuisance and its
    penalty floats it at zero.

  Both `a` and `t` are per-observation latent fields — exactly the
  case `LatentCoord` (`latent_coord.md` §3) generalises from one
  to two sub-blocks.

- **ρ** = `(λ_sp, λ_sm, {α_kj}_{k,j}, K)` where `λ_sp` is the
  sparsity strength on `a`, `λ_sm` is the smoothness strength on
  each `B_k`, `α_kj` is the ARD precision on the `j`th axis of
  atom `k`'s latent coordinate, and `K` is the discrete atom count.
  Continuous components of ρ are selected by REML in the usual
  outer loop; `K` is selected by `compare_models`
  (`gamfit/_compare.py:97`) over candidate atom counts.

### 3.2 Reconstruction and loss

Each atom `k` has a smooth decoder `g_k : ℝ^{d_k} → ℝ^p` of the form
`g_k(t) = Φ_k(t) · B_k`, where `Φ_k` is a basis on atom `k`'s
internal latent manifold (a `Duchon`, `Sphere`, `PeriodicSplineCurve`,
or tensor-product basis, exactly as in any other gamfit smooth). The
reconstruction is the assignment-weighted sum of decoder outputs:

```
Ẑ_i = Σ_k a_ik · g_k(t_ik) = Σ_k a_ik · Φ_k(t_ik) · B_k.
```

The full penalised-likelihood objective is

```
ℒ(β, ψ, ρ) =
    ½ Σ_i ‖Z_i − Ẑ_i‖²                                 (data fit)
  + λ_sp Σ_i Pen_sparse(a_i)                            (atom sparsity)
  + λ_sm Σ_k B_k' P_k B_k                               (decoder smoothness)
  + Σ_k Σ_j α_kj ‖t_{·, k, j}‖²                         (ARD per atom × axis)
  + R_id(a, t)                                          (identifiability; §3.4)
  − ½ log |Σ(ρ, ψ)|                                     (REML log-det).
```

`Pen_sparse` is the existing active-set L¹ (`src/solver/active_set.rs`)
on the non-negative orthant; `P_k` is the same Duchon-style
function-norm penalty already used by every smooth (`Duchon` doc at
`gamfit/smooth.py:79`); the ARD term is the `dim_selection="reml"`
construction from `latent_coord.md` §2.3(c) replicated per atom.

### 3.3 Tier assignment of the gradient

All gradients are analytic and reuse existing machinery:

- `∂ℒ/∂B_k` is the standard penalised-LS normal-equation block.
- `∂ℒ/∂a_ik` = `−(Z_i − Ẑ_i)' g_k(t_ik) + λ_sp · ∂Pen_sparse/∂a_ik`;
  the data-fit term is the inner product of the residual with
  atom `k`'s decoded value at `t_ik`. This is row-local in `i`,
  exactly the `LatentCoord` row-local structure.
- `∂ℒ/∂t_ik` = `−a_ik · (Z_i − Ẑ_i)' (∂Φ_k/∂t · B_k)`; the
  `∂Φ_k/∂t` factor is the radial-kernel chain rule already supplied
  by `ImplicitDesignPsiDerivative` (`src/terms/basis.rs:3948`).
- `∂ℒ/∂ρ` is the standard REML gradient with the additional
  ARD-per-axis terms; the implicit-function-theorem propagation
  through the inner solution is the existing
  `evaluate_unified_with_psi_ext` path.

No autograd. No new IFT scaffolding. The arrow / bordered-Hessian
structure is exactly the structure `composition_engine.md` §6
identifies for `LatentCoord`: per-observation ψ blocks are
block-diagonal across `i`, coupled to the shared β only through the
dense border, and the Schur complement reduces the per-step cost from
`O((Nd+M)^3)` to `O(N d^3 + M^3 + Nd · M)`.

**Audit caveat (REML `log|H|` gradient).** The arrow shape holds at
the cost level; the gradient of the REML Occam term has a shared
`Schur⁻¹` factor — it is *not* literally a sum of independent per-row
contributions. Specifically, `∂/∂t_i log|Schur| =
tr(Schur⁻¹ · ∂Schur/∂t_i)` with `Schur⁻¹` dense in all `t`, but
`∂Schur/∂t_i` is a rank-≤d update because only row `i` of `Φ` moves
with `t_i`. So per outer iteration: one dense `Schur⁻¹` formation
(shared) plus N cheap per-row traces. Asymptotic cost is
`O(N + decoder-cost)` per outer iteration, not `O(N²)`. See
`latent_coord.md` §2.2 for the full derivation.

### 3.4 Identifiability priors

Three gauge issues exist and all three are addressed by primitives
already proposed:

1. **Atom-label permutation.** Trivial and harmless; collapses on
   `compare_models` by atom-set equivalence.
2. **On-atom reparameterisation** (the GP-LVM gauge of
   `latent_coord.md` §2.3). Resolved by the `IsometryToReference`
   penalty (`composition_engine.md` §4(b)) or by an auxiliary-conditional
   prior on `t_ik`. ARD on the `t_ik` axes is a useful *companion* — it
   identifies which axes carry signal once gauge is fixed — but per the
   math audit it is *not* a standalone gauge fix: `α_j ‖t_{·,j}‖²` is
   rotation-symmetric, so ARD alone can re-shuffle "used" vs "unused"
   axes without changing the objective. Combine ARD with the isometry
   or aux-prior; do not rely on ARD on its own.
3. **Atom overlap.** When two atoms' decoded sets overlap in
   ℝ^p, the assignment `a_ik` is non-unique on the overlap. The
   conditional-prior route from `composition_engine.md` §4(c) —
   `R_id(a, t) = ½ τ ‖a_i − g_φ(Z_i)‖²` with `g_φ` a small
   amortised encoder — handles this; equivalently a hard
   winner-take-all assignment as a simplex projection collapses
   the orbit at the cost of differentiability through the
   assignment. See §8 for the choice.

## 4. Why the existing engine already covers this

The composition is mechanical:

- **β extends penalised-LS.** A multi-atom decoder is a multi-output
  smooth on a block-diagonal design `diag(Φ_1, …, Φ_K)` weighted
  per-row by `a_ik`. The `by`-multiplier path on `Smooth`
  (`gamfit/smooth.py:73`) already accepts per-row multipliers on a
  smooth's contribution. The new content is making `by` be one of the
  ψ blocks rather than a frozen observed column.

- **ψ is the LatentCoord block with two sub-fields.** The
  `LatentCoord` spec proposed in `latent_coord.md` §3 already
  packages a per-row latent vector with explicit `id_mode`. The
  extension here is structural: the latent field for SAE-manifold
  has two named sub-blocks (`a` of shape `(N, K)`, `t` of shape
  `(N, K, d_k)`), both flowing through the same
  `HyperDesignDerivative` plumbing. The `is_penalty_like = false`
  semantics (`src/solver/reml/mod.rs:2704`) already isolate the
  design-moving case correctly.

- **ρ is the existing REML outer loop plus discrete K.** Continuous
  ρ components (`λ_sp`, `λ_sm`, `α_kj`) are selected by the
  existing `RemlState::evaluate_unified_with_psi_ext` path. The
  discrete atom-count `K` is selected by `compare_models`
  (`gamfit/_compare.py:97`) over fits at K ∈ {K_min, …, K_max},
  comparing REML scores. This is the same recipe used for
  `topology`-selection cited in `composition_engine.md` §5.

- **Active-set inner solver handles sparsity on `a`.** The existing
  active-set machinery (`src/solver/active_set.rs`) is built to do
  exactly this; sparse-coding is one of its named use cases in the
  composition-engine table (row "Mechanism / sparse-coding amplitude
  gating").

- **Arrow / bordered-Hessian structure.** Per-observation ψ
  (`a_i`, `t_i·`) is block-diagonal across `i`, coupled to the
  shared `B = (B_1, …, B_K)` border only through the dense
  cross-block. Schur-eliminate β first (the existing inner
  factorisation), then solve the row-local `(a, t)` blocks; this is
  the same pattern that makes the standard `LatentCoord` workflow
  viable for large `N`. (Caveat: the REML `log|H|` gradient carries
  a shared `Schur⁻¹` factor — see §3.3 above and `latent_coord.md`
  §2.2. Per-row contributions decompose into rank-≤d traces against
  a Schur inverse computed once per outer iteration; the asymptotic
  cost stays linear in `N`.)

In short: every piece is either already shipping
(`compare_models`, active-set, `BlockwisePenalty`, REML outer loop,
`HyperDesignDerivative`) or already proposed (`LatentCoord` and
its `id_mode`).

## 5. The shattering fix, formal

### 5.1 The problem

A linear SAE represents an input cloud living near a `d_C`-dimensional
curved manifold `C ⊂ ℝ^p` as a dictionary of `K` atoms `D_k ∈ ℝ^p`
with sparse non-negative codes. Because `D_k` enters linearly, each
atom can only represent a single direction; a curved 1-D feature
(canonical example: the hue circle of a colour-sensitive layer) is
expressed by tiling `K` linear atoms along it. The atoms are mutually
correlated, individually uninterpretable ("hue ≈ 0.42 atom"), and the
dictionary count `K` scales as `O(M/r)` where `M` is the arclength of
`C` and `r` is the linear resolution at which each atom is locally
faithful. This is the standard "shattering" or "feature splitting"
pathology.

### 5.2 The fix in one line

Replace the `K` linear atoms with a single **manifold-atom** whose
decoder is a smooth `g : ℝ^{d_C} → ℝ^p`. The manifold-atom's
parameters are a single `B ∈ ℝ^{M × p}` plus a per-observation
on-atom coordinate `t_i ∈ ℝ^{d_C}`. Reconstruction is `Ẑ_i = a_i ·
Φ(t_i) B`. This is the K=1 special case of §3.

### 5.3 Why REML may prefer the manifold over the shards

Earlier drafts presented an unambiguous determinant proof that REML
"decisively" picks the manifold over `K` linear shards. The math audit
flagged that sketch as overstated. The defensible statement is:

> With properly normalised priors, fixed model families, controlled
> gauge, comparable fit, and smooth-decoder effective dimension growing
> slower than the required linear-shard count, marginal likelihood may
> favour the manifold beyond some `K_required`. This is not an
> unambiguous determinant proof; the asymptotic direction depends on
> the specific prior normalisation and decoder complexity.

Cast informally: the linear-K-atom model carries a log-evidence Occam
load that scales with `K` (one decoder block and one sparsity-code df
contribution per atom), while the manifold-atom model carries
`df_man ≈ M · p + N · d_C` whose first term is REML-selected via
`λ_sm` and whose second is bounded by ARD. Provided the prior
normalisers (`(N/2) log α`-style terms, smoothness determinants) are
all included and gauges are pinned, the manifold side's Occam load
grows sublinearly in `K_required` while the linear side's grows
linearly. That motivates — but does not prove — a crossover beyond
some `K_required`. Sign and threshold depend on `p`, `d_C`, the
normalisation choices, and the realised decoder complexity. The
empirical hook in §5.4 is what currently carries the load.

### 5.4 Empirical hook

This is not just an algebraic argument. The
`/Users/user/Manifold-SAE` Curve-SAE benchmark already realises a
prototype of this on synthetic data:

> On pure-curve data (D=256, 16 GT smooth curves, 32 positions each,
> 30K samples): per-active-atom, the curve SAE is ≈3.7× more
> efficient than vanilla TopK SAE, with 4× fewer features needed for
> matched MSE, and zero dead atoms. (Source: project memory
> `project_curve_sae_efficiency.md`.)

That prototype is hand-rolled; the proposal here is the gamfit-native
re-expression of the same finding, with REML doing the work that the
manual Curve-SAE benchmark left to grid search.

## 6. API sketch

The user-facing entry point is a thin wrapper that constructs the
right `LatentCoord` blocks and pipes them into `gamfit.fit`. Style
matches the existing `gamfit.fit` (`gamfit/_api.py:604`) and the
`Smooth` hierarchy at `gamfit/smooth.py:33`.

```python
import gamfit
from gamfit import LatentCoord  # from latent_coord.md §3

# Direct construction — explicit.
model = gamfit.fit(
    data=dict(Z=Z),                          # Z shape (N, p)
    formula="Z ~ s(t, basis='duchon', m=2, by=a) - 1",
    latents={
        "a": LatentCoord(
            n=N, d=K_atoms,                  # one nonneg amplitude per atom
            init="kmeans",                   # warm-start from k-means on Z
            constraint="nonneg",
            sparsity_prior="l1",             # active-set L1 on a
            sparsity_strength="auto",        # REML over λ_sp
        ),
        "t": LatentCoord(
            n=N, d=K_atoms * d_per_atom,
            layout=("per_atom", K_atoms, d_per_atom),
            ard=True,                        # ARD per (atom, axis)
            init="pca-per-cluster",
        ),
    },
)
```

A higher-level wrapper for the common case:

```python
result = gamfit.sae_manifold_fit(
    Z,                                       # (N, p) activations
    n_atoms=10,                              # int, or "auto" — Occam-selected
    atom_dim=None,                           # int, or "auto" via ARD
    atom_basis="duchon",                     # or per-atom: list of Smooth specs
    sparsity_strength="auto",                # REML over λ_sp
    smoothness="auto",                       # REML over λ_sm
    init="kmeans",
)

# Returns a list of fitted atoms; each carries decoder, basis, and
# the per-observation assignment + on-atom coord.
for atom in result.atoms:
    B_k             = atom.decoder_coefficients   # (M_k, p)
    basis_k         = atom.basis                  # Smooth spec
    a_per_obs       = atom.assignments            # (N,) nonneg
    t_per_obs       = atom.coords                 # (N, d_k); valid where a > 0
    reml_score      = atom.evidence
```

Internally `sae_manifold_fit` does:

1. If `n_atoms="auto"`: candidate list `K ∈ {K_min, …, K_max}` and
   `compare_models` over their REML scores
   (`gamfit/_compare.py:97`).
2. If `atom_dim="auto"`: each atom's `LatentCoord` is constructed
   with the per-axis ARD `dim_selection="reml"` flag from
   `latent_coord.md` §2.3(c).
3. Per-atom topology: `atom_basis` may be a single string applied
   uniformly, or a list aligning with `K` — each entry a `Smooth`
   spec (`Duchon`, `Sphere`, `PeriodicSplineCurve`, …) so that
   `topology(atom_k)` is a model choice not a fit choice.

## 7. Implementation status

Already in flight or shipping:

- **`compare_models`** (`gamfit/_compare.py:97`) — picks `K`.
- **REML outer loop** — picks `λ_sp`, `λ_sm`, `α_kj`.
- **Active-set inner solver** (`src/solver/active_set.rs`) —
  enforces non-negativity and sparsity on `a`.
- **`HyperDesignDerivative` / `evaluate_unified_with_psi_ext`** —
  propagates IFT through design-moving ψ.
- **`by`-multiplier on `Smooth`** (`gamfit/smooth.py:73`) — the
  multiplicative gate for `a` × decoder.
- **Topology wrappers** — `Duchon`, `Sphere`,
  `PeriodicSplineCurve`, `TensorBSpline` (`gamfit/smooth.py:79+`).
- **`LatentCoord` proposal** (`latent_coord.md`) — the per-row
  latent field; the spec used here.
- **Penalty library (in flight by parallel agent)** — isometry,
  sparsity, ARD-per-axis. The SAE-manifold work consumes these as
  black-box primitives, no implementation work duplicated here.
- **Geodesic-acceleration Newton patch** (`composition_engine.md`
  §4(e)) — the nonlinear-residual problem of SAE-manifold is
  exactly the regime where this matters; default it on for this
  configuration.

What remains, specific to SAE-manifold:

1. **Multi-atom dispatch glue.** The fit assembly needs to register
   `K` parallel smooth terms, each with its own per-atom basis and
   its own slice of the `a` and `t` `LatentCoord` blocks. The
   `latents={}` mapping in `gamfit.fit` is the natural seam; the
   work is the formula-side syntax for per-atom-distinct bases.
2. **Evidence-based selection of K.** Wrapper code over
   `compare_models`; no new core machinery, but a sensible default
   schedule of candidate `K` values (geometric ladder seeded by
   `kmeans` BIC, capped by `K_max`).
3. **Layout-aware `LatentCoord`** — extend the `LatentCoord` shape
   so that the (N, K, d) tensor's `K` axis is treated as
   block-diagonal with the assignment `a` gating it; this is the
   "two sub-blocks" generalisation of §3.1, and the Rust-side change
   is small (the storage `Array1<f64>` already supports a
   `dims_per_term`-style layout).
4. **`gamfit.sae_manifold_fit` Python entry point.** Thin wrapper.

## 8. Open questions for maintainers

**Assignment parameterisation.** Three options for `a_i ∈ ℝ^K_{≥0}`:
(a) softmax of free logits — keeps the simplex, breaks pure sparsity;
(b) non-negative free amplitudes with active-set L¹ — supports exact
sparsity, lets the total scale float (which is fine in the
reconstruction objective and can be regularised by the existing
double-penalty); (c) simplex projection — exact sparsity and exact
totalling, non-smooth gradient at the boundary. Default proposal: (b),
because it falls straight into the existing active-set inner solver
and matches the standard TopK-SAE convention.

**Atom overlap.** When two atoms' decoder images overlap in ℝ^p, `a`
is non-unique on the overlap. The auxiliary-conditional-prior
construction from `composition_engine.md` §4(c) with a small amortised
encoder `g_φ(Z_i)` is one route; a hard winner-take-all assignment
gated by `compare_models` is another. Maintainers' preference?

**Initialisation.** `kmeans` on raw `Z` to seed `K` cluster centres,
then `pca` per cluster to seed `t_ik`, then 1 epoch of decoder fit at
warm-started `(a, t)` to seed `B_k`. Anything beyond this (e.g. a
larger TopK SAE pre-trained, then collapsed into manifold atoms) is
a research direction not a default.

**Per-atom topology API.** Should `atom_basis` accept a callable
`Z_cluster → Smooth` so the basis choice can be data-driven (e.g.
"if persistent cohomology says `H^1 ≠ 0`, use
`PeriodicSplineCurve`, else `Duchon`")? This is the cleanest hook for
the TDA-driven workflow that `composition_engine.md` §5 sketches.

**Per-atom `d_k` mixed dimensions.** With `atom_dim="auto"` and ARD,
different atoms may converge to different intrinsic dimensions. The
storage layout `LatentCoord` (`Array1<f64>` + `dims_per_term`) already
supports ragged per-term dim; expose this or force `d_k = d_max` for
all atoms with ARD pruning? Probably ragged once the layout work
in §7 lands.

**Identifiability minimum.** `composition_engine.md` §7 takes the
position that `LatentCoord` should refuse to fit absent at least one
gauge-breaking choice (aux prior, isometry). Per the math audit, ARD
is **not** on that list: `α_j ‖t_{·,j}‖²` is rotation-symmetric and
does not by itself break the on-atom reparameterisation gauge. The
SAE-manifold configuration has ARD on `t` and L¹ on `a`; the L¹ on
non-negative `a` does pin atom-amplitude scaling, but it does not pin
the on-atom coordinate gauge. Recommendation: require either
`IsometryToReference` per atom or an aux-conditional prior tying
`t_ik` to an observed `u_i` whenever `d_k > 0`. Worth confirming with
maintainers.

---

This is one configuration of one engine. Naming it makes the
shattering pathology stop looking like a feature of SAEs and start
looking like what it actually is: a basis-mismatch error, fixed by
giving the basis enough expressive power to track the manifold.

## 9. Audit revisions

This document was revised in response to a math-audit pass on the
original optimistic draft. Tightened claims:

- **§1 premise.** "REML Occam factor decisively prefers one curved
  atom over `K` linear shards" softened to "can prefer," with the
  caveat list cross-referenced to §5.3.
- **§3.3 tier-assignment of the gradient.** Added an audit caveat
  describing the shared `Schur⁻¹` factor in the REML `log|H|`
  gradient. The arrow/Schur cost still holds, but per-row
  contributions are not literally independent — one Schur inverse is
  formed per outer iteration and N rank-≤d per-row traces are
  evaluated against it.
- **§3.4 identifiability priors (item 2 on-atom reparameterisation).**
  Removed the claim that ARD on `t_ik` axes is itself a gauge fix.
  ARD is a useful *companion* to `IsometryToReference` or
  aux-conditional priors; it is not a substitute.
- **§4 "Why the existing engine already covers this"
  (arrow/Schur bullet).** Added the audit caveat about the shared
  `Schur⁻¹` factor.
- **§5.3 "Why REML picks the manifold over the shards" → "Why REML
  may prefer the manifold over the shards".** The determinant proof
  sketch was wrong as written and has been replaced with the
  defensible verbatim form from the audit: marginal likelihood may
  favour the manifold beyond some `K_required` *given* proper prior
  normalisation, fixed model families, controlled gauge, comparable
  fit, and smooth-decoder effective dim growing slower than the
  required linear-shard count. The empirical Curve-SAE 3.7×
  efficiency result (§5.4) is now the load-bearing argument; the
  theoretical sketch motivates rather than proves.
- **§8 identifiability minimum open question.** ARD removed from the
  list of acceptable standalone gauge fixes; explicit recommendation
  added that `IsometryToReference` per atom or an aux-conditional
  prior be required whenever `d_k > 0`.

Source: math-audit findings summarised in
`/Users/user/.claude/projects/-Users-user-Manifold-SAE/memory/project_gamfit_composition_engine.md`
(section "Math-audit findings").
