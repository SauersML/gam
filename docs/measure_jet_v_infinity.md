# Measure-jet spline V∞ — the exact realization (charter + seam map)

The estimand is FROZEN: the analysis-form multiscale jet-residual energy of
the empirical measure documented in `src/terms/basis/measure_jet_smooth.rs`,
with its contracts (exact constant annihilation, τ-ridged rank-adaptive
jets, Mellin band, density normalization, frozen-quadrature replay). V∞
changes nothing about the object. It deletes V0's three approximations —
the function space (Gaussian representers), the quadrature (center-mass
collapse), and the algebra (dense m×m, dials at seeds) — and computes the
same estimand exactly, fast, with the honesty contract intact.

Every component below is mapped to machinery this repo already maintains.
One moment-table source feeds the design products, the prior, every ψ-jet,
the support curve, and the spectrum report (single-source rule, elevated).

## 1. Function space: the jet frame is the model

- Coefficients = multiscale innovations on the per-level ε/2-nets that
  `assemble_weighted_forms` already constructs as outer quadrature; the nets
  are PROMOTED to index sets of the basis.
- Atom = Gaussian bump at scale ε_ℓ × local jet monomials {1, frame
  coordinates}; coarse-to-fine polynomial PREDICTION (lifting) so level ℓ
  carries only what coarser jets fail to predict (vanishing moments by
  construction).
- Unpenalized global polynomial block (degree < r) in the arrow HEAD —
  exact ambient-affine pass-through at ANY τ: the ridge prices innovations,
  never the trend. Kills the O(τ) affine toll structurally.
- Prior diagonal by coordinates: independent whitened innovations,
  per-level variances λ_ℓ⁻¹ (the per-scale-candidate mode made structural).
- License: frame equivalence A·J_s ≤ Σ_ℓ ε_ℓ^{−2s}‖d_ℓ‖² ≤ B·J_s, with A, B
  ESTIMATED at runtime (Lanczos probes on the whitened operator) — every
  fit ships its own frame-ratio certificate.

## 2. Data interface: moments or nothing

- The only computation over the n rows: Gaussian-weighted moment tables per
  net cell per level — coordinate orders 0–2 (order 2(r−1) general) crossed
  with channels {1, y, y², PIRLS working z, w}.
- Merge law = binomial shift μ′_α = Σ_{β≤α} C(α,β)(c−c′)^{α−β} μ_β: an
  associative, commutative, deterministic monoid ⇒ exact distributed
  fitting, exact online updates, bit-reproducibility under sorted
  reduction.
- All Gram entries, XᵀWX products, ψ-jets, and support-curve values are
  closed-form Hermite couplings of stored moments. The model and the fast
  Gauss transform are one object; the only approximation is the truncation
  radius with its explicit e^{−ρ²/2} bound charged to the tolerance budget.
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
- NEW: learned anisotropy A = LLᵀ as a ψ-block (Hermite-derivative moments,
  one pass per step); per-coordinate τ_k = (σ_{x,k}/ε)² unifying coordinate
  noise, quantization, and the rank-adaptive ridge — the formal license for
  low-precision moment inputs.
- Nets/masses/frames stay x-only and frozen (honesty trichotomy); optional
  bootstrap over net seeds folds into reported geometry variance.

## 5. Distance-honest prediction (the V0 honesty bug, fixed structurally)

V0's representers decay off-support toward the parametric backbone with
SMALL Vp — confident reversion, which the contract forbids. V∞ predictive
at x★ = jet extension from the first covering scale ε★(x★) (read off the
support curve already computed from the frozen model) + closed-form
extrapolation variance Var_extrap(x★) = Σ_{ℓ: ε_ℓ ≥ ε★} λ̂_ℓ⁻¹ a_ℓ(x★).
Microseconds, no solve; intervals widen monotonically with distance from
the web because the same fitted spectrum that smooths on-support prices
ignorance off-support. Support label + band + interval become one
statement.

## 6. Order and junctions

- r = 2 default (order-2 ambient moments: d(d+1)/2 per cell).
- r = 3 via the two-pass trick: frames from order-2 moments, then moments
  of frame-projected coordinates to order 4 in q ≤ 8 dims; frame-transport
  mismatch at merges bounded and charged to the certificate budget.
- Junctions: nothing new — ridge-jets make vertex behavior a smooth
  crossover; nets keep all-arm cells at stars so shared-jet coupling
  emerges.

## 7. Acceptance gates (all existing gates kept verbatim; these are added)

1. Exact affine pass-through at DEFAULT τ (basis property; unit test exits
   oracle mode).
2. Off-support variance growth: Vp + Var_extrap monotone in distance from
   the web.
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
   monoid + closed-form Gram couplings, oracle-tested against direct
   assembly. (No consumer change yet; the substrate.)
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
