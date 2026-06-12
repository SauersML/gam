# Measure-jet spline V∞ — charter + seam map

The estimand is FROZEN: the analysis-form multiscale jet-residual energy of
the empirical measure documented in `src/terms/basis/measure_jet_smooth.rs`,
with its contracts (exact constant and affine annihilation, rank-revealing
local affine projection, Mellin band, density normalization,
frozen-quadrature replay). V∞ changes the realization, not the displayed
analysis-form target: it replaces V0's Gaussian-representer function space,
mass-lumped quadrature, and dense algebra with controlled frame, quadrature,
and sparse-factorization approximations that carry explicit certificates.

Every component below is mapped to machinery this repo already maintains.
Moment tables feed exactly the frozen-weight polynomial couplings they store;
moving Gaussian transforms such as support curves and Gaussian Gram products
are separate kernel evaluations or controlled approximations.

## 1. Function space: the jet frame is the model

- Coefficients = multiscale innovations on the per-level ε/2-nets that
  `assemble_weighted_forms` already constructs as mass-lumped outer
  quadrature; the nets are PROMOTED to index sets of the basis.
- Atom = Gaussian bump at scale ε_ℓ × local jet monomials {1, frame
  coordinates}; coarse-to-fine polynomial PREDICTION (lifting) so level ℓ
  carries only what coarser jets fail to predict (vanishing moments by
  construction).
- Unpenalized global polynomial block (degree < r) in the arrow HEAD —
  exact ambient-affine pass-through: penalties price innovations, never the
  trend. Kills the old O(τ) affine toll structurally.
- Prior diagonal by coordinates: independent whitened innovations,
  per-level variances λ_ℓ⁻¹ (the per-scale-candidate mode made structural).
- License: frame equivalence A·J_s ≤ Σ_ℓ ε_ℓ^{−2s}‖d_ℓ‖² ≤ B·J_s, with A, B
  ESTIMATED at runtime (Lanczos probes on the whitened operator) — every
  fit ships its own frame-ratio certificate.

## 2. Data interface: moments or nothing

- The row-streaming substrate builds Gaussian-weighted moment tables per net
  cell per level — coordinate orders 0–2 (order 2(r−1) general) crossed with
  channels {1, y, y², PIRLS working z, w} — for the kernel centers and scales
  actually requested.
- Merge law = binomial shift μ′_α = Σ_{β≤α} C(α,β)(c−c′)^{α−β} μ_β: an
  associative, commutative, deterministic monoid for frozen-weight
  polynomial moments ⇒ exact distributed accumulation and
  bit-reproducibility under sorted reduction. It re-expresses `(x-c)^α`
  under the same weights; it does not move the Gaussian kernel center.
- Local polynomial Gram blocks, `XᵀWX` products under frozen weights, and
  same-center ψ channels are closed-form couplings of the stored moments.
  Support curves, Gaussian Gram entries at other queries, and Gaussian
  `XᵀWX` products with moved kernels require their own kernel pass or a
  certified approximation; order-2 moments alone cannot determine them.
- Repo seams: third moment substrate sibling to `gpu/cubic_cell`
  (host_substrate / kernel_src / device NVRTC layout) and the `bms`
  chunked-row-reduction streaming pattern. CPU streaming reference lands
  first (`measure_jet_moments.rs`); the GPU sibling follows the existing
  NVRTC pattern.

## 3. Solver: one factorization, everything exact

- Coefficients Σ_ℓ |net_ℓ| (geometric across levels; finest level capped at
  cells holding ≥ q+1 points) — ~10⁶–10⁷ at n = 10⁸, never n.
- Whitened M = I + D^{1/2}SᵀWS D^{1/2} is arrow-shaped: dense
  polynomial/parametric head + level-banded tail with bounded stencils —
  exactly `solver/arrow_schur.rs`'s structure, including its per-row
  rank-≤d trace machinery for REML log|H| gradients.
- Two certified modes: coarse-deflated CG at FIXED iteration count (legal
  because conditioning = the measured frame ratio), or supernodal sparse
  Cholesky in coarse-to-fine net order (elimination tree = net tree;
  screening fill bound): pivots = exact log-det, selected inverse = exact
  trace gradients, perturb-and-optimize = exact posterior draws.
- Spectral loop with ZERO data passes: ∂M/∂log λ_ℓ = −(level-ℓ block of
  M − I) ⇒ all per-level amplitudes, ŝ, σ², and the double-penalty dial via
  level-blocked traces off one factorization. Geometry steps (metric, τ)
  are the only data-touching loop: one moment pass each, derivative
  channels fused (Hermite-weighted Gaussians).

## 4. Dials

- (s, α, lnτ) jets: shipped, FD-gated, consumed by the live ψ enrollment.
- Density normalization: on a p-dimensional stratum with sampling density
  `ρ`, `q_ε ~ Cρ ε^p` and the local affine residual scales as
  `R_ε ~ Cρ ε^{p+4}|Hf|²`, so the limiting energy carries density
  `ρ^(3−2α)`. The current `α = 1` default is density-weighted Hessian
  energy; density-free Hessian energy would use `α = 3/2`.
- NEW: learned anisotropy A = LLᵀ as a ψ-block (Hermite-derivative moments,
  one pass per step); per-coordinate noise scales can feed the
  rank-revealing projection threshold and certificate budget — the formal
  license for low-precision moment inputs without adding an affine ridge.
- Nets/masses/frames stay x-only and frozen (honesty trichotomy); optional
  bootstrap over net seeds folds into reported geometry variance.
- Center/barycenter collapse and the ε/2 outer net are first-moment-exact
  mass lumping. Gaussian functionals are not preserved identically; their
  relative scale is controlled by the cell diameter through
  `O(diam²/ε²)` when the kernel is smooth on the cell.

## 5. Distance-honest prediction (the V0 honesty bug, fixed structurally)

V0's representers decay off-support toward the parametric backbone with
SMALL Vp — confident reversion, which the contract forbids. V∞ predictive
at x★ = jet extension from the first covering scale ε★(x★) (read off the
support curve already computed from the frozen model) + closed-form
extrapolation variance. With `q̄_ℓ` the frozen on-web support mean,
`a_ℓ(x★)=min(q_ℓ(x★)/q̄_ℓ, 1)`, and
`ℓ★=min{ℓ : q_ℓ(x★) ≥ floor·q̄_ℓ}`,

```text
Var_extrap(x★) =
  Σ_{ℓ < ℓ★} λ̂_ℓ⁻¹ + Σ_{ℓ ≥ ℓ★} (1 − a_ℓ(x★)) λ̂_ℓ⁻¹.
```

This is monotone under pointwise support domination: if one query has
`q_ℓ` no smaller at every scale, its extrapolation variance is no larger.
Ordinary monotonicity in Euclidean distance from the web is false in
general; a bimodal support distribution can put an on-center point at full
variance while a between-mode point has lower variance. Support label, band,
and interval become one statement.

## 6. Order and junctions

- r = 2 default (order-2 ambient moments: d(d+1)/2 per cell).
- r = 3 via the two-pass trick: frames from order-2 moments, then moments
  of frame-projected coordinates to order 4 in q ≤ 8 dims; frame-transport
  mismatch at merges bounded and charged to the certificate budget.
- Junctions: nothing new — ridge-jets make vertex behavior a smooth
  crossover; nets keep all-arm cells at stars so shared-jet coupling
  emerges.

## 7. Acceptance gates (all existing gates kept verbatim; these are added)

1. Exact affine pass-through at the default settings.
2. Off-support variance growth: Vp + Var_extrap obeys the support-domination
   theorem above; plain distance monotonicity is not a valid gate.
3. Near-miss strand decoupling: parallel strands at separation δ share no
   value coupling at affine order (estimand-level; landable against V0).
4. Peak-in-the-gap under r = 3 (curvature carried across a hole).
5. Scale smoke: n = 10⁶–10⁷ build+fit wall-clock and bytes asserted in CI —
   "fast" as a gate, not a vibe.
6. Certificates printed per fit: truncation, frame ratio,
   refine-one-level delta, backward error — tolerances budgeted at
   δ·min_T se(T).

## 8. Slice order (each lands tree-consistent)

1. `measure_jet_moments.rs`: CPU streaming moment tables + binomial-shift
   monoid + closed-form frozen-weight polynomial couplings, oracle-tested
   against direct assembly. (No consumer change yet; the substrate.)
2. Estimand-level acceptance gates that don't wait for the frame basis
   (near-miss decoupling; scale smoke on the V0 path).
3. §5 extrapolation-variance seam: pure function + predict-side wiring
   plan; fuse when the frame basis lands.
4. Jet-frame basis mode (per-level nets → atoms, lifting, unpenalized
   head) behind the same `mjs()` surface; per-level candidates become the
   structural diagonal; V0 representer path retired the moment the frame
   path passes every existing gate (no parallel layers kept).
5. Arrow/Schur whitened solver wiring; spectral identities; certificates.
6. GPU moment substrate (NVRTC sibling); A-metric ψ-block; r = 3.
7. Manifold-SAE hybrid: frozen x-only `latent_coord`/SAE frames into the
   metric slot; evidence adjudicates the learned chart.
