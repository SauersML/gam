## v0.3.146 — gam 0.3.146 / gamfit 0.1.248 (2026-07-03)

Correctness and honesty release on top of 0.3.145, cut from a large batch of
root-cause fixes. Two themes run through it. First, **honest reporting**: the
solver now surfaces non-convergence, line-search stalls, and genuine
rank-deficiency instead of papering over them, a NaN regression in the reported
Gaussian log-likelihood/AIC is fixed, and a new SPEC rule forbids wall-clock
time budgets and deadlines (all of which have been removed from the solver).
Second, **no arbitrary knobs**: remaining grid scans and magic constants are
replaced by principled searches — a golden-section search for the Tweedie
variance power, a monotone-bisection ψ-band search, and error-driven
order-doubling for cloglog GHQ quadrature. Alongside are new `loglog`/`cauchit`
survival links, weight-aware analytic Gaussian observation intervals, several
model-persistence and class-detection repairs on the Python surface, a
differentiable-basis (Duchon) autodiff correctness fix, and a broad round of
work on the experimental, default-off SAE manifold engine.

### Solver honesty & policy
- **No wall-clock time budgets or deadlines (#2055).** SPEC now forbids
  wall-clock time budgets/deadlines, and every such budget/deadline has been
  removed from the solver rather than left to paper over a slow path. Related
  survival-path deadline guards were removed or scoped, dropping the last
  banned underscore-let bindings introduced to silence them (#979).
- **Report non-convergence honestly (#2062).** The flexible-link inner-KKT
  envelope no longer masks a non-converged inner solve as converged; it reports
  the failure. Custom-family exact-joint nonconvergence stays recoverable
  (#2014) instead of aborting the fit.
- **Multinomial line-search stalls are reported, not hidden (#2066).** A stalled
  line search no longer returns `converged = true`.
- **Identifiability audits tell the truth (#2070).** The audit reports genuine
  rank-deficiency honestly; rank-restore keeps structural aliases dropped and
  restores only numerical demotions, and structured-residual fit failures
  propagate instead of being swallowed.

### Families & links
- **`loglog` and `cauchit` survival links.** Both links are implemented for the
  survival family and covered across the `LinkFunction` scale/routing paths,
  completing #1946; their GHQ routing replaces a banned `unreachable!()` with an
  explicit branch.
- **cloglog GHQ quadrature converges by construction (#2063, #1835).** cloglog
  Gauss–Hermite quadrature now certifies convergence via error-driven
  order-doubling around an adaptive mode-centred rule, instead of a fixed order
  that could silently under-resolve.
- **Automatic Tweedie power without a grid (#2064).** The bare-`family="tweedie"`
  variance-power estimate (added in 0.3.145) now uses a golden-section search
  over the open interval `(1, 2)` in place of the coarse power grid scan.
- **SAS / beta-logistic link fixes (#1876, #2094).** The SAS ε sign convention is
  corrected and value/cost consistency restored with regression tests (dropping
  a banned `#[allow]`), and the SAS/beta-logistic link spec is now threaded
  through the formula and Python fit paths.

### Estimation & inference
- **Gaussian log-likelihood/AIC NaN regression fixed (#2096).** The reported
  `log_likelihood` (and the AIC derived from it) came out NaN for every Gaussian
  fit because the profiled scale was passed through unresolved; the profiled
  σ̂² is now resolved into the reporting spec while the persisted scale marker
  stays `ProfiledGaussian`. Covered by a finiteness regression over the real
  reporting assembler.
- **Weight-aware analytic Gaussian observation intervals (#2077).** For a
  weighted Gaussian fit, `predict(observation_interval=True)` now scales the
  per-row band by `1/√wᵢ`, matching `sample_replicates` (the analytic sibling of
  #2025) instead of broadcasting a single pooled σ̂².
- **Multinomial Firth/Jeffreys separation solver (#1854, #1821).** A
  self-contained fixed-λ Firth/Jeffreys solver handles separation; Firth stays
  active in the LM line-search candidate and gate-motion is added to the
  Jeffreys inner gradient.
- **Location-scale reproducibility (#1607).** Cross-fit warm-start is gated on a
  family opt-in so exact-replay parity holds; the binomial location-scale outer
  gradient is aligned with finite differences.
- **Identifiability ridges (#2068, #1802).** `double_penalty` linear terms get
  exactly one identifiable ridge (no duplicate null-space ridge), and the
  reparam null-leakage tolerance is derived from the eigensolver PSD floor
  rather than a fixed constant.
- **ψ-band search de-gridded (#2054).** The rank-stable ψ-band search uses
  monotone bisection instead of a grid.
- **Thin-plate spline basis recovery (#1966, #1074).** Fixes low-rank
  under-recovery by seeding the length-scale from center spacing.
- Tier-0 rho-posterior certificate is emitted on formula fits (#1810);
  `partial_dependence` by-factor blocks no longer depend on input row order;
  matrix-free tangent-projected logdet operator trace in REML (perf).

### Python surface (gamfit)
- **Multinomial-logit GAM persistence (#2078).** Multinomial-logit models now
  round-trip through the public `save`/`load` API.
- **`is_marginal_slope` for Bernoulli marginal-slope models.** The kebab-case
  `marginal-slope` model_kind label is now recognized, so `is_marginal_slope`
  returns `True` for Bernoulli marginal-slope fits.
- **`GAMClassifier` non-{0,1} labels (#2075).** The response column is dropped at
  inference for classifiers with non-`{0,1}` labels.
- **`Model.model_class` precision.** Derived from the fine-grained predict class
  rather than the coarse `model_kind` enum; `Model.evidence` /
  `bayes_factor_vs` route through the `compare_models` ranking score (#2079).
- Prior weights preserved in smooth-significance; stagewise progress callback
  types exported.

### Differentiable bases (torch)
- **Duchon basis VJP correctness (#2097).** `_DuchonBasisFn.forward` now sources
  its design from the same jet builder that backward differentiates, so the
  analytic input gradient is the exact derivative of the returned forward
  (max|analytic−fd| ~3e-11, down from ~0.7) and the width-mismatch case that
  raised a broadcast `RuntimeError` no longer occurs. gradcheck/gradgradcheck
  regressions added.

### SAE manifold engine (experimental, default-off)
- Stagewise births are unblocked on disjoint/near-block-diagonal residuals via a
  Marchenko–Pastur-thresholded residual-principal fallback, with anchor-scored
  birth-seed selection (#2080); topology-seed subsample/kNN/interp are derived
  from budget and dimension (#2065); total co-collapse bails with a diagnostic
  (#2089).
- The certified encode path now reads the true (un-ridged) Hessian for the
  Kantorovich certificate, with amplitude-aware routing and Duchon refusal.
- New public surface: coordinate interchange and coordinate-fidelity APIs
  (#2019, #2069), stagewise birth checkpoints exposed through the Python FFI, and
  GPU dictionary scoring wired with route telemetry.
- Router throughput: an O(1)-reject top-`s` fold and a shared-memory-tiled score
  GEMM at the `K ≈ 32k` dictionary width (#1026, perf).

### Build & CI
- The `sparse_dict_router_topk` bench and the `sae` integration test are fixed to
  the post-carve module path / signature so `cargo check --all-targets` links.
- The `build.rs` author guard is robust to shallow clones; an on-demand
  single-job "Validate One" workflow gives fast per-issue fixer feedback.

## v0.3.145 — gam 0.3.145 / gamfit 0.1.247 (2026-07-02)

Correctness release on top of 0.3.144. The headline is a prior-weights
effective-sample-size cluster that makes zero-weight and zero-`by` rows exactly
inert in Gaussian REML, plus mgcv-style automatic Tweedie variance-power
estimation for a bare `family="tweedie"`. Alongside are a multinomial-Firth
log-determinant consistency fix, a binomial location-scale predict-noise repair,
a Duchon well-posedness auto-raise, and constrained-REML backward/forward
corrections. This cut also removes leftover FD-audit debug scaffolding from the
κ-optimization path and hardens the (experimental, default-off) SAE dictionary
diff so it can no longer report "no differences" when atoms were added or
removed.

### Families & responses
- **Automatic Tweedie variance power for bare `family="tweedie"` (#2026).**
  Mirroring mgcv's `tw()`, a bare `family="tweedie"`/`"tw"` (CLI `--family
  tweedie`, or a formula family that names no explicit power) now estimates the
  variance power `p ∈ (1, 2)` by profile likelihood before the reported fit — a
  coarse grid over the open interval refined by golden-section search, profiling
  the dispersion out per node with the solver's own prior-weighted Pearson
  estimator and scoring the fully-normalized saddlepoint log-likelihood
  (comparable across `p`). Previously a bare Tweedie silently used a fixed
  `p = 1.5`, miscalibrating observation intervals on data whose true power was
  not 1.5. An explicit `tweedie(1.6)` / `tweedie(p=1.6)` still pins `p` exactly.
- **Weighted replicate frames keep their weight column (#2033).** Regression of
  #2025: `Model.sample_replicates` narrowed the frame to the required prediction
  columns + response before resolving per-row weights, so every weighted model
  raised `weights column '…' not found in data`. The weight column is now part of
  the consumable-column set (so it survives projection), and the replicate path
  degrades to unit weights when the caller frame genuinely omits it instead of
  erroring. Ordinary predict is unaffected.

### REML / prior weights
- **Zero-`by` rows are inert in Gaussian REML (#2031).** `by` was applied only
  as a design-column gate, so a `by=0` row's response still entered the response
  energy `Σ w·y²` and the residual degrees of freedom `ν`, leaking into `σ²`,
  `λ`, and — through `λ` — the coefficients. The `by` gate is now folded into the
  REML row weights (`w_eff = w·[by≠0]`) across every forward/backward FFI path, so
  a `by=0` row is a complete no-op. A fit whose `by` is entirely nonzero stays
  byte-identical to manually gating the design.
- **Residual DoF uses the effective sample size (#2032).** Zero prior-weight
  rows (the universal "excluded / infinite-variance" convention) no longer count
  toward `ν = n − nullity`. `ν` is now built from the number of strictly
  positive-weight rows everywhere the REML score, `σ²`, `λ`, `edf`, and the
  adjoint are computed, so a `weights=0` row is exactly equivalent to omitting it.
  When every weight is positive this is a strict no-op.
- **Constrained REML forward/backward (Rust CI / Python API).** The constrained
  forward now returns the closed-form Gaussian REML solve whenever the optimum is
  interior — both when no inequality system is supplied *and* when a system is
  present but the unconstrained optimum is already strictly feasible (every
  `aᵢ·β̂ > bᵢ`, the exact negation of the active-set binding test). By the KKT
  conditions an interior certificate is the unconstrained problem, so this makes
  a non-binding constraint agree bit-for-bit with the unconstrained
  `gaussian_reml_fit` and with the interior-cert backward (which already
  differentiates that closed form) instead of settling on a slightly different
  PIRLS smoothing parameter. Binding (shape-constrained) fits are unchanged. The
  backward's zero-bound guard is additionally scoped to the *active* face, so an
  interior certificate carrying a non-zero bound on a never-binding slack
  constraint (e.g. `0·β ≥ −1`) flows through the correct full-space envelope VJP
  instead of being rejected up front.

### Inference & prediction
- **Multinomial-Firth log-determinant consistency (#1854/#1395).** The
  small-system `BlockCoupledOperator` route eigendecomposes via
  `eigh(Side::Lower)`, which assumes a symmetric input; on the near-separating
  Firth path the divided-difference curvature carries an `O(1e10)` scale, so
  reduction-order floating-point asymmetry produced a materially different
  spectrum and log-determinant than the #1395 ground-truth guard (which
  symmetrizes first). The joint Hessian is now symmetrized before the operator is
  built, so every route realizes the penalized joint Hessian log-det
  consistently.
- **Binomial location-scale predict-noise (#1828).** The default-link arm no
  longer demands an explicit blend spec (it falls back to the binomial-logit base
  link), and a parametric-linear `log_sigma` is accepted for predict-noise while
  a nonparametric free scale is still correctly rejected.
- **BMS marginal / log-slope blocks lock to raw width (Rust CI / Python API).**
  The Bernoulli marginal-slope and log-slope Jacobian callbacks now declare
  `locks_raw_width_reduction()`, mirroring the survival marginal-slope precedent,
  so the canonicaliser keeps their raw block width.

### Smooths & kernels
- **Duchon null-space order auto-raises to clear the collocation margin
  (#1817).** A low order/power pair with a derivative-collocation operator active
  (e.g. `d=2`, `Linear` null space, `power=0` with the stiffness operator) could
  trip the pointwise/collocation well-posedness guard `2(p+s) > d + max_op`
  mid-fit. The null-space order `p` is now auto-raised by the smallest amount that
  restores the strict margin before the guard can fire (warned once per config);
  the spectral power and the CPD condition `2s < d` are untouched.
- **κ design-realization skip restored on the n-free lane (#1868/#1033).** The
  `TEMP-SKIPOFF-1122` debug override that hard-forced the `O(n)` lane in
  `eval_full` is removed, restoring the n-free gradient/value path. The remaining
  leftover FD-audit debug scaffolding on that path (four unconditional
  `TEMP-*-1122` blocks that logged at warn level and rebuilt the Matérn basis /
  penalty triplet several times per call) is deleted.
- **Cubic-cell C0-continuity regression made meaningful (#1837).** The
  `bug_hunt` continuity check was mis-specified — it built the neighbouring cell
  from a cell-local Taylor parameterization but stored those coefficients as a
  global polynomial, and compared the two cells at `boundary ± eps` against a
  tolerance tighter than the injected `O(eps·slope)` gap. The test now constructs
  the right cell through the kernel's own `global_cubic_from_local` path and
  evaluates both cells at the shared boundary point, so it is a genuine (passing)
  C0 invariant check. The production kernel is unchanged.

### SAE manifold (experimental, all default-off)
- **Chart-transfer operators (#2016).** Pulled-back chart-to-chart transfer
  operators `A_kj(x) = (JₖᵀJₖ)⁻¹Jₖᵀ J_F(x) Jⱼ(x)` for 1-D/2-D atoms, with
  density-weighted mean/variance aggregation (Kish effective-n) and
  isometry/equivariance transport certificates. (The coordinate-valued
  attribution-graph deliverable is scaffolded but not yet wired.)
- **Canonical dictionary artifact tooling (#2018).** Deterministic SHA-256
  hashing of the Frobenius-normalized, reflection-fixed dictionary orbit
  representative, with order/scale/reflection-invariant equality and a
  decoder-row-localizing diff. The alignment diff now counts atoms with no
  counterpart on either side as substantive differences, so
  `hash_equal_after_alignment` can no longer claim equivalence when the two
  dictionaries carry different atom sets at equal total count. (Residual
  continuous-chart gauge pinning and Procrustes/optimal-transport alignment
  remain future work.)
- **Sparse SAE Schur block GEMM (#1995).** The reduced-Schur block subtract skips
  zero columns on the sparse atom support.

### Testing
- Rust and Python regression coverage for the zero-weight / zero-`by` REML
  inertness (#2031/#2032), the estimated-Tweedie-power recovery (#2026), the
  Duchon auto-raise (#1817), the interior-cert non-zero-slack-bound constrained
  backward, the generic row-kernel jet oracle projections (#932), and the SAE
  dictionary diff's unmatched-atom accounting.

## v0.3.144 — gam 0.3.144 / gamfit 0.1.246 (2026-07-02)

Correctness release on top of 0.3.143. Two new user-facing capabilities land on
the formula/CLI surface — the Tweedie variance power is now settable and the
`loglog`/`cauchit` survival links are wired end-to-end — alongside a batch of
prediction/diagnostics fixes (Gaussian location-scale σ scale double-count, a
`gam diagnose` failure on every standard fit, and generative replicate noise
that ignored prior weights). This cut also repairs a release-blocking cubic-cell
regression that landed after 0.3.143: a well-meaning affine-anchor "normalization"
turned five previously-green deep-tail/both-tails precision guards red, and is
reverted here to the raw substrate convention the whole kernel (and the CPU/GPU
parity reference) actually uses.

### Families & responses
- **Tweedie variance power on the formula path (#2026).** `family="tweedie(1.6)"`
  / `tweedie(p=1.6)` now parse the mgcv-style parenthesized argument as the
  variance power `p` (`Var = φ·μ^p`) and validate it through the shared
  strict-`(1, 2)` gate, instead of misrouting `1.6` to the link resolver and
  failing with `unknown link '1.6'`. A non-numeric argument (`tweedie(log)`)
  still flows to the link resolver; bare `tweedie`/`tw` keep the neutral interior
  default `p = 1.5`.
- **Survival `--link loglog` / `--link cauchit` (#1829).** Both links are now
  accepted and evaluate exactly, routed through a single-component mixture
  (weight 1.0, no free mixing logits) so they flow end-to-end through the wired
  `InverseLink::Mixture` survival path; the survival `--link` usage string
  advertises them. Genuine multi-component blends still require a
  logit/probit/cloglog anchor.
- **Weighted Gaussian replicate noise (#2025).** `Model.sample_replicates` now
  scales each row's Gaussian observation noise by its analytic prior weight
  (`σ_i = σ̂/√w_i`, `Var(y_i) = σ²/w_i`) instead of a single pooled scalar.
  Unit/absent weights leave unweighted fits unchanged.

### Inference, prediction & diagnostics
- **Gaussian location-scale predict σ no longer double-counts `response_scale`
  (#1874, #1928).** The persisted log-σ intercept is already shifted by
  `+ln(response_scale)` at fit time, so only the soft floor (which sits outside
  the `exp`) is scaled at predict time — exactly one factor of `response_scale`
  on the σ surface, restoring response-scale equivariance whenever
  `sample_std(y) ≠ 1`.
- **`gam diagnose` fixed on every standard fit (#2030).** Batch compaction no
  longer zeroes the row-sized `working_weights`/`working_response` on the
  persisted geometry carrier, so the geometry-ALO path stops handing empty
  vectors to `AloInput::from_geometry` and failing length-N validation; a
  present-but-emptied carrier now falls through to the refit branch, and the
  saved weight column is loaded into the diagnose frame.
- **Distinct penalty coordinates for grouped + double-penalty + block-gamma
  priors (#1881).** Combining coefficient groups, per-term double penalties, and
  keyed block-gamma priors now keeps each as its own λ instead of collapsing the
  per-term base/double coordinates into one shared linear ridge.

### Smooths & kernels
- **Cubic-cell affine-anchor moments kept in the raw substrate convention
  (#1833).** Reverts the post-0.3.143 normalization of the public
  `affine_anchor_moment_vector` (dividing by `√(2π)`), which broke the #352
  both-tails and deep-tail precision guards (relative error `1 − 1/√(2π)`) and
  diverged the public API from every production consumer and the byte-for-byte
  CPU/GPU parity reference. The mis-specified identity test is corrected to the
  raw standard-normal moments (`M0 = M2 = √(2π)`, `M1 = M3 = 0`).
- **Duchon affine-trend native ridge is curvature-relative (#880).** The
  machine-scale ridge on the affine slope columns is now scaled by the curvature
  block's mean diagonal (genuinely `√ε`-relative, as documented) rather than an
  absolute floor that survived Frobenius normalization and pushed the affine
  trend out of the null space on low-magnitude curvature grids.

### SAE manifold (experimental, all default-off)
- Two-tier fit primitives (tier merge / atom reorder), the scale-gauge quotient
  with no-refresh amplitude-absorb transport-peel, data-row dead-atom reseed, and
  the Λ nursery→promotion birth channel are promoted from hidden `GAM_SAE_*` env
  levers to typed, default-`false` kwargs (`promote_from_residual`,
  `quotient_scale`, `data_row_reseed`), threaded through the pyffi entry points
  including the IBP convenience delegator (#2021, #2022, #2023). The historical
  default path stays bit-for-bit unchanged.

### REML / ALO internals
- FD gates added for the Firth and Barrier first-order `D_βH` curvature
  operators; the ALO sandwich-SE meat weight is taken from the score/Fisher
  weight `w_s` rather than the observed-information weight `w_h`, with a
  defense-in-depth reject of non-positive `snr_proxy` in the SAE global-optimality
  verdict.



Broad correctness, inference-accuracy, and performance release on top of
0.3.142. The headline changes: grouped-**binomial proportion responses** are
accepted again (not just strict `{0, 1}`); a batch of REML / posterior-variance
/ credible-band fixes tighten uncertainty quantification; the Duchon and Matérn
spline families get several geometry and null-space corrections; the manifold
SAE stack lands device-resident execution, principled (no-magic-constant)
penalties, and a batched JumpReLU gate kernel exposed to `gamfit.torch`; and a
wave of PERF work brings 1-D/2-D smooth and location-scale fits from
2–160× slower-than-reference down toward parity. This cut also repaired three
release-blocking compile regressions introduced by concurrent landings (a
clobbered `ReportInput.smoothing_forensics` field in the PyO3 wheel, a stale
`ArrowBlockDiagInverse::build` call site, and an unbalanced-delimiter test) and
removed a banned `#[allow(too_many_arguments)]` by bundling the JumpReLU layout
inputs into a params struct.

### Families & responses
- **Binomial proportion responses accepted (#1806, #1987).** The binomial
  support was tightened to strict `{0, 1}` in an earlier cut, which rejected
  grouped-binomial and continuous-probability responses (with trial weights
  folded into the row weight). Support is now the closed interval `[0, 1]`,
  which keeps the Bernoulli / grouped-binomial log-likelihood bounded; the
  strict `{0, 1}` predicate is retained only where the code is specifically
  asking whether a numeric response is binary (auto-inference and all-boundary
  degeneracy checks). External GLM response-support errors are also clarified
  (#1982).
- **Survival lognormal formula routing fixed (#2009);** reduced-AFT MLE
  non-convergence is handled gracefully (#1921); KKT projection includes
  near-active monotone rows to clear a flat baseline-hazard stall (#1793,
  #1992); left-truncated transformation LAML uses a hard-pseudo logdet (#1915);
  all-interval latent survival warm start restored (#1916); survival constraint
  projection semantics clarified (#1919).
- **Beta-Logistic** derived delta now uses the SAS bounded transform (#1993);
  negative-binomial dispersion location-scale ρ optimization fixed (#1956);
  Gaussian location-scale fits retain their joint covariance (#1974) and restore
  σ to response units on generate (#1928).

### Inference, uncertainty & REML
- **Logit posterior-variance regression fixed (#1976);** smooth credible bands
  get a corrected ρ-variance cubature gate (#1971) and a bias-correction
  linearization on the smoothing-corrected β covariance (#1970); the ALO
  sandwich SE is computed from the frozen-curvature meat `XᵀWX` (#1969).
- **Hutchinson REML trace standard-error scaling corrected (#1977);** REML folds
  and B-spline double-penalty EDF inflation get deterministic row reductions and
  a curvature-only cleanup (#1929, #1964); single-ρ̂ Lawley mean-shift uses
  Gauss–Hermite quadrature (#1972); `Gamma(1, 0)` precision priors are treated as
  `Flat` for the REML ρ-prior policy (#2006).
- **Random-effect BLUP recovery fixed (#2008);** the summary penalty cursor now
  skips only the penalty blocks the random effects actually own (unpenalized
  `by`-factor ranges no longer slide every following smooth's window off by one)
  (#1883, #1979); concurvity-driven double-penalty null-space rail collapse is
  guarded (#2017); the marginal-slope absorber is protected from
  over-orthogonalization with an `n`-scaled ridge (#2013, #1947).
- **Multinomial per-class smoothing recovery fixed (#1855, #2027);**
  identifiability audit gauge attribution and rank-deficiency handling fixed
  (#2005); Royston–Parmar uncertainty prediction link canonicalization fixed
  (#1955); prediction CSV schema stabilized (#1928).

### Smooths & bases
- **Duchon family:** rotation-equivariant knot tie-breaks (#1935), affine-trend
  ridge deselection to prevent pre-fit rank deficiency (#1934), polynomial
  null-space evaluation in the collocation operators (#1933), and nullspace-test
  centering (#1931); periodic Duchon auto-power pinned to `s = 0`.
- **Matérn:** sufficient basis centers for exact adaptive regularization (#2003);
  anisotropic derivative geometry — normalization, center RRQR, η seeding (#1937).
- **Cubic-regression (`cr`/`cs`)** bases routed to the standard dense fit off the
  exact spline-scan fast path, with a `ds` Duchon alias added (#1844, #1957);
  cyclic B-spline seam evaluation fixed (#1922) and small-k cyclic tensor margins
  allowed (#1944); tensor margins honor quantile knot placement (#1943);
  `by`-factor / `bs="sz"` `SmoothBasisSpec` construction fixed (#1981);
  constant-curvature `curv()` recovers hyperbolic data instead of railing to
  spherical (#1464).

### Manifold SAE
- **Batched JumpReLU gate FFI kernel (#6).** `gamfit.torch`'s bounded threshold
  gate is now a thin marshaller over a single batched Rust value+grad call
  (`sae_jumprelu_batch_value_grad`), bit-identical to the row kernel — the whole
  `(N, K)` matrix crosses the Python↔Rust boundary once per forward pass instead
  of once per row, and Rust is the single source of truth for the gate math.
- **Device-resident SAE solver engages on production fits (#1017, #1551);** the
  GPU offload gate is tuned for thin `d_atom = 1` curve atoms (#1913) and the
  throughput decision gate is made honest against its target (#1412).
- **Principled penalties (#1610):** hand-picked magic-constant penalty strengths
  and collapse thresholds replaced with derived quantities; per-fit IBP α
  override on manifold init avoids atom masking (#1784); IBP cross-row Woodbury
  coupling restored (#1920); SAE evidence Schur deflation uses unit-stiffness
  quotient conditioning for the log-det (#1925); recoverable refusal probes kept
  finite (#1912). The JumpReLU compact layout is sized by the hard forward gate
  (O(k_active)), and the residual-factor path hoists row-independent work into a
  reusable `fit_row_metric` (#2021 Wave-2). A `WhitenedStructured` row-metric
  driver is wired onto the fit path behind a default-off flag (#2021).

### Performance
- 1-D P-spline / thin-plate and Poisson/Gamma `s(x)` fits (#1689, #1690),
  binomial-logit `s(x)` / `te(x, z)` and REML fits (#1575, #1727),
  Gaussian-location-scale (`noise_formula`) overhead (#1720), and Duchon 2-D
  fits (#1718, #1757) are all brought substantially closer to reference-software
  speed at equal accuracy. GAM significance vs reference software improved
  (#1561).

### Reports, CLI & reliability
- **Smoothing-forensics report section added** (λ/σ² paths, EDF criterion vs
  assembly) (#1986). `--family expectile` and other standard-family CLI fits no
  longer abort on the frailty guard in the expectile inner fit (#1948).
- Concurrent tail-cell cache computations are coalesced to avoid duplicate work
  (#2002); warm-start LRU touch and deterministic eviction fixed (#1998);
  power-law scaling reports hardened against noisy timings (#1997); the spatial-κ
  optimizer recovers from a NaN frequentist covariance by treating the trial
  point as infeasible (#2001); startup no longer bails on repeated non-finite
  trial objectives (#1802, #1924).

## v0.3.142 — gam 0.3.142 / gamfit 0.1.244 (2026-07-01)

Broad correctness, robustness, and performance release on top of 0.3.141. The
headline fixes: left-truncated survival fits stop collapsing to a degenerate,
covariate-flat curve; explicit `--family` and frailty flags stop misreporting or
aborting; `periodic=false` on a spatial smooth is honored; the isotropic-Matérn
location-scale ψψ Hessian is no longer silently halved; several previously
panicking or misleading edge cases now return clean errors; and a batch of
solver/GPU/SAE performance work lands with reproducibility and matrix-free
correctness guards.

### Survival
- **Left-truncated `Surv(entry, exit, event)` no longer degenerate (#1790,
  #1791).** Under the default `transformation` (Royston–Parmar) likelihood, any
  delayed entry (`entry > 0`) produced a covariate-independent fit with cumulative
  hazard inflated ~10³× and `S(t) ≡ 0`. Two root causes are fixed: (a) the time
  basis is now anchored at the robust interior median exit under genuine left
  truncation (not the earliest entry, whose one-signed centered linear-trend
  column railed the smoothing selection), extending the marginal-slope #751 fix to
  every time-basis likelihood; and (b) because the transformation LAML uses the
  **observed** information (which the delayed-entry `−Xᵀ_entryW_entryX_entry` block
  can drive indefinite below the seed λ), the time smoothing blocks' outer lower
  bound is floored at their seed ρ under left truncation, so the selector may only
  hold or over-smooth the baseline — never rail it into the degenerate
  under-smoothed region. Right-censored (`entry == 0`) fits are bit-for-bit
  unchanged. Regression tests pin covariate recovery and baseline non-degeneracy.
- **Survival predict-query times accept the full real line (#965).** The
  predict-query coercion (`survival_at`/`hazard_at`/`cumulative_hazard_at`/CIF)
  now rejects only NaN; `S(t≤0)=1`, `S(+∞)=0`, `H(+∞)=+∞` are threaded through the
  interpolation and CSV paths (distinct from the finite past-grid flat-clamp), and
  a latent `right_value` inconsistency in the chunked path is fixed. Rust + Python
  regressions added.

### Families & CLI
- **Explicit `--family` no longer prints a false inferred-family note (#1781).**
  The "Inferred …-family" note was gated only on `link_choice.is_none()`, so e.g.
  `--family gamma-log` reported "Inferred gaussian-identity family" while fitting
  and saving a Gamma/log model. It is now additionally gated on the family being
  `Auto`, so the data-heuristic note fires only when auto-discovery actually ran.
- **`--family expectile` (and any standard-family CLI fit) no longer aborts on a
  null frailty spec (#1780).** The CLI always populates `frailty =
  Some(FrailtySpec::None)`; the standard/survival/transformation guards tested
  `Option::is_some()` and so misread that canonical "no frailty" value as a
  frailty request. Every guard now tests `FrailtySpec::is_active`, eliminating the
  whole class of null-`Some` misreads; genuine frailty requests are still rejected
  where unsupported.
- **Beta external-family rejection points at Binomial GLM routing (#1888).** The
  unsupported-Beta-response message now notes that a binary `{0,1}` response is a
  Binomial GLM and should be routed through the Binomial family.

### Smooths & terms
- **`periodic=false` on a spatial smooth builds a NON-periodic basis (#1676).**
  The scalar-boolean shortcut returned `Some([None])` for `false`, which the
  radial 1-D consumers read as "periodicity requested, derive the wrap from the
  data range" — so an explicit `periodic=false` silently produced a *periodic*
  smooth. It now returns `None`, matching the bracketed `[false]` form; covered
  for matern/thinplate/duchon. (`scale_dimensions` / boolean-periodic acceptance
  for thin-plate also landed under #1676.)
- **Isotropic-Matérn ψψ second-design derivative restored in the
  transformation-normal operator (#1607).** The matrix-free
  `TensorKroneckerPsiOperator` gated `∂²X/∂ψ²` on `implicit_group_id.is_some()`,
  which is `None` for an isotropic single-length-scale Matérn — so its ψψ diagonal
  was silently dropped to zero, halving the outer ψψ Hessian / LAML curvature and
  the Firth/Jeffreys term on those fits. The gate now routes the isotropic
  self-second-derivative through the operator (the same fix #1607 applied to the
  custom-family resolver), keyed on the global covariate-deriv index so distinct
  ungrouped blocks never synthesise a spurious cross term.

### Fitting & inference correctness
- **Double-penalty nullspace shrinkage escapes the inflated-EDF trap (#1266).**
  The pure-REML shrink-out path now reaches the bending coordinate and selects
  unsupported terms out, so `s(x)` on linear data shrinks toward its true EDF
  instead of railing high.
- **`debiased_functional(target="point"/"contrast")` no longer errors when the
  response column precedes the predictor (#1621).** Bookkeeping (placeholder)
  columns are lenient-encoded in the query design, mirroring predict's
  frame-projection, so the query feature index is no longer out of bounds.

### SAE (sparse dictionary)
- **Massive-K dictionaries fit matrix-free (#1026)** off the streaming
  arrow-factor cache, with a collapsed-linear-lane minibatch made load-bearing via
  batched-GEMM routing and overcomplete `K ≥ n` admitted under identifiability;
  overcomplete `K ≫ p` periodic atoms get a generic diverse seed (#1893).
- **Recoverable infeasible-ρ Schur-seed refusal presents as a finite wall (#1782)**
  in every outer lane instead of a fatal startup abort; the spectral PD-floor is
  wired into the inner/log-det solves (#1038); the off-diagonal-only IBP cross-row
  Woodbury θ-adjoint is corrected (#1416); born-atom topology races score by proper
  REML (#977); `ρ.log_lambda_smooth` grows with K on birth/fission (#1556);
  `random_state` is honored in the closed-form fast paths (#178); the
  `assignment_prior`/`n_atoms` kwarg aliases have a real contract (#159/#160); and
  explicit-ψ Firth/Jeffreys terms are gated on Jeffreys-information ψ-dependence
  (#901).

### Performance
- **Duchon 2-D fit reuses the un-rotated design (#1718)** instead of rebuilding
  it, fusing the reparam rotation into a single GEMM.
- **Binomial-logit REML Firth outer-Hessian cost cut (#1575)** by fanning the
  per-penalty direction/pair loops across Rayon (index-ordered, bit-reproducible)
  and bounding the TK-Hessian mixed-matvec cache.
- **Per-eval ALO leverage diagnostic skipped via a ρ-independent non-activation
  certificate (#1689)**; default solver logging quieted to `Warn` with an env-free
  `--log-level` (#1688).
- **Device-resident SAE joint fit** engagement, sphere-Wahba GPU kernel, fused
  arrow-Schur block strides, and NVRTC FMA-contraction CPU parity land with
  fail-loud false-routing guards and V100-verified parity (#1017 and related).

### Robustness / error handling
- **Empty designs are rejected cleanly (#1848).** `fit_gam` guards `n == 0` /
  `p == 0` at the entry point with an `InvalidConfig` error instead of panicking on
  zero-sized indexing / linear solves.
- **Device CSR construction enforces `rowptr.len() == rows + 1` (#1846)** via a
  canonicalizing `DeviceCsrMatrix::new`, closing an invalid-free / out-of-bounds
  deallocation hazard.

### Also in this release
- Survival lognormal fits request the location-scale route so they are dispatched
  correctly (#1847).
- Multinomial separation auto-engages the Firth/Jeffreys bias-reduction fallback
  (#1854).
- `summary` penalty-cursor skips unpenalized by-factor random-effect ranges so the
  per-term penalty accounting lines up (#1883).
- Scale-invariant rank tolerance in `positive_spectral_whitener_from_gram` (#1889).
- Real REML-2.3 solver-invariant tests (#1861).

### Release hygiene
- `trace_product_sparse` (`tr(H⁻¹S)`, feeding the REML gradient/EDF) now reduces
  its per-column partials serially in column order, so the result is bit-identical
  across rayon thread-pool sizes rather than drifting in the low bits with core
  count (#759); pinned by a 1-vs-8-worker regression.
- Dead scaffolding removed from shipping code: the vestigial `resolve_log_level` /
  `default_log_level` indirection and a misleading env-override doc in the
  env-free logging path (#1688/#1689), and the orphaned #178 Python LCG-jitter
  constants + a stale comment in the `gamfit` wheel source. A Duchon-fusion test
  header that overclaimed "bit-for-bit" is corrected to the tolerance it asserts.

## v0.3.141 — gam 0.3.141 / gamfit 0.1.243 (2026-07-01)

Correctness patch on top of 0.3.140, focused on non-converged / near-degenerate
fits that previously returned an unusable model or contradicted their own reported
effective degrees of freedom, plus a matrix-free path that lets massive-K SAE
dictionaries descend their hyperparameters instead of hard-erroring.

### Fitting / inference correctness
- **Non-converged estimated-scale fits return a USABLE model (#1789).** When the
  #1788 EDF-collapse guard re-derives the dispersion `σ̂² = RSS/(n − edf)` after
  correcting a collapsed effective d.f., it now rescales BOTH redundant covariance
  representations — the top-level `covariance_conditional`/`covariance_corrected`
  AND the paired inference-block `beta_covariance*`/SEs — atomically through the
  new `UnifiedFitResult::rescale_estimated_dispersion`. Previously it scaled only
  the inference block, so a non-converged multi-smooth gamma / `[INDEF-HESS]` fit
  returned a `Model` that `fit` accepted but `predict`/`summary`/`save`→`load` all
  rejected with "inference conditional covariance must match top-level
  covariance_conditional". The rescale can no longer touch one copy and not the
  other, and its `#[must_use]` σ̂ ratio is now reported in the non-convergence
  warning instead of being discarded.
- **Self-contradictory penalized EDF on stalled REML fits is guarded (#1788).** A
  non-converged fit whose influence EDF collapsed to the intercept-only floor while
  its fitted coefficients stayed wiggly now substitutes the per-term dimension
  floor (so the reported EDF is not self-contradictory) and surfaces the
  non-convergence rather than shipping a silent collapse.
- **Firth fallback rescues binomial-logit REML near-separation stalls (#1762).** A
  binomial-logit fit that stalls in the flat REML valley near quasi-separation is
  retried once with Firth/Jeffreys bias reduction and adopts that result only if it
  actually converges, otherwise preserving the honest base result/error.
- **Log-link PIRLS enforces shape-constraint feasibility (#1786).** Monotone/convex
  smooths on low-count Poisson (log-link) no longer silently ship coefficients that
  violate the requested shape cone: an LM-damping retry precedes a
  feasibility-restoring projection (curvature re-evaluated at the projected β), and
  an infeasible fit errors rather than returning an invalid model.
- **Massive-K SAE descends its hyperparameters matrix-free (#1026).** The EFS
  (Fellner–Schall) lane now takes its ARD/smoothness traces off the streaming
  arrow-factor cache returned by `reml_criterion_streaming_exact_with_cache`
  instead of forcing the dense `O((K·M·p)²)` evidence cache that hard-errors at
  large K (25.9 GB even at K=256). The `gamfit` facade admits an overcomplete
  dictionary (`K ≥ n`) under ARD/smoothness-prior identifiability — a warning, not
  a refusal.

### CLI
- **`gam predict --uncertainty` keeps the posterior-mean point for curved links
  (#1787).** The point-estimate column is the response-scale posterior mean for
  curved-link families, matching the Python FFI, instead of being swapped to the
  linear-predictor mean when `--uncertainty` was requested.
- **Prediction-CSV linear-predictor column header restored to `eta`.** The base,
  Gaussian location-scale, survival, and survival-binary CSV writers had drifted
  to emitting `linear_predictor`; the schema-lock contract (and every downstream
  reader) expects `eta`. All four writers now emit `eta` again. The Python FFI
  dict-key contract (`linear_predictor`) is a separate path and is unchanged.

### Python / docs
- **transformation-normal `predict` output documented as E[Y|x] (#1612)**, not a
  z-score.
- **Constrained-REML active-set recompute (gam-pyffi)** partitions rows by KKT
  activity (`a·β ≤ b + tol`), not by feasibility, so interior rows are no longer
  spuriously reported active.

### Test / build hygiene
- Restored the `gam-sae` test binary: an #1784 IBP-capacity refactor left an
  `Option`/`Result` mismatch (compile break) and stale large-scale assertion
  margins; margins are recalibrated to the RAM-safe scale and a regression test now
  pins the #1026 streaming cache as a drop-in for the dense cache in the EFS lane.
  A closed-form-criterion bench also no longer discards a `#[must_use]` solve
  `Result`.

## v0.3.140 — gam 0.3.140 / gamfit 0.1.242 (2026-07-01)

Release-integrity and correctness patch on top of 0.3.139. The headline is that
the tree **builds and packages again from a clean checkout**: the 0.3.139
release shipped a workspace that aborted its own build (build.rs hygiene-ban
violations), so `cargo build`/`test`, a fresh `--release`, and the `maturin`
wheel all failed on a cold `target/`. Every violation is cleared and pinned with
regression guards, on top of a batch of root-cause fixes: a mixed-boundary
tensor smooth that hard-errored, the SAE IBP-MAP high-noise stall, and a GPU
BMS-FLEX row-kernel parity bug.

### Fitting / inference correctness
- **Mixed periodic + clamped tensor smooths fit instead of hard-erroring.** A
  tensor margin's non-periodic spelling `clamped`/`open` (the B-spline-clamped,
  free-ended margin) is now accepted in the `boundary=`/`bc=` list, so
  `te(theta, z, boundary=['periodic','clamped'], period=[2*pi, None])` — gam's
  analog of mgcv `te(bs=c("cc","ps"))` for a cylinder — builds a cyclic θ margin
  tensor-producted with an ordinary open z margin. Previously the guard rejected
  `clamped` as an unsupported endpoint reparameterization, taking out the
  cylinder / solar-zenith / cyclic-tensor recoveries with an IntegrationFailed.
  A genuine `anchored` zero-value endpoint constraint (no ordinary-margin
  meaning on a tensor) is still surfaced as a clean unsupported-feature error.
- **SAE IBP-MAP reaches a flexible fit at high noise instead of stalling (#1744).**
  Two root causes are repaired: (1) the IBP-MAP ρ seed is no longer
  response-dispersion-scaled — that Gaussian-normal-equation identity is invalid
  for IBP's free Bernoulli gates and let the inner solve overfit at the seed,
  collapsing the Fellner–Schall fixed point to zero penalty; and (2) the
  parsimonious keep-best no longer lets a *less-stationary, more-smoothed*
  non-converged seed displace the flexible incumbent on a marginally-lower
  non-converged REML alone. Together the planted-circle IBP fit reaches EV ≈ 0.95
  at σ=0.18 instead of stalling at 0.86.

### GPU / numerical correctness locks
- **BMS-FLEX GPU row kernel uses the observed predictor VALUE for the probit
  Mills margin (#415).** The device kernel and its host oracle were reading
  `bar_e_u[0]` — the u=0 first-derivative jet — as `e_obs` instead of the
  degree-0 value `η(a(θ),θ;z_obs)`, diverging from the CPU family's
  `signed_margin = s_y·eta_val`. The observed value is now packed and consumed
  directly, locked by a non-vacuous CPU-oracle == CPU-family parity test over
  every row's value, full gradient, and full r×r Hessian.
- **Survival I-spline time-penalty PSD invariant is locked at construction
  (#979).** The value-space curvature penalty is assembled as the full
  congruence `Lᵀ S L` and *then* reduced to the kept columns (PSD by
  construction); a regression test with a non-trivial `keep_cols` asserts the
  assembled penalty is PSD, so a future reassembly that reintroduces an
  indefinite reduction is caught at construction rather than as a silent
  outer-loop hang.

### Build / release integrity
- **The workspace builds from a clean checkout again.** Cleared the build.rs ban
  violations that shipped onto the 0.3.139 release: a temporary exploratory
  coverage probe committed into `src/` (stdout prints + a `#[cfg(test)]` module
  dodge), `construction.rs` over the 10k-line tracked-file limit, a `#[ignore]`
  test, and a stale `uv.lock` gamfit version.
- **The #415 CPU-oracle test module stays private.** Its cross-file consumer was
  relocated into a private sibling test module so the oracle is reached in-module
  — no `pub(crate)` on a `#[cfg(test)] mod`, which the ban gate (correctly)
  rejects and which had re-broken the build.
- **Removed the always-`panic!` #1765 observation-coverage probe.** It was
  inconclusive exploratory scaffolding — its own fixed-seed numbers show the
  residual-df scale is well-calibrated on that ridge sweep and `edf2` is not the
  lever — so it neither reproduced the bug nor validated a fix; the finding is
  recorded on the issue and the Gaussian scale stays guarded by
  `gaussian_high_edf_scale_tests`.
- **The `sigma_link` source-pattern guard no longer self-trips** — it skipped a
  stale `families/sigma_link.rs` path and so scanned its own literal, failing the
  CI test shard on every commit.
- Added fast unit locks for the tensor boundary-token guard and made the 0139
  bug-hunt regression test robust to future version bumps.

Everything reachable through the existing API stays backward-compatible.

## v0.3.139 — gam 0.3.139 / gamfit 0.1.241 (2026-07-01)

crates.io + PyPI release of the post-0.3.138 wave. The headline is the SAE /
manifold stack moving its fit and encode paths onto Rust and the `gamfit`
package thinning to a SPEC-compliant wrapper (numeric math lives in Rust), on
top of a batch of root-cause correctness fixes, two new pieces of public surface
(periodic radial builders and the expectile/LAWS family reachable from both the
Python API and the CLI), and GPU/perf work proven bit-identical to the scalar
path. Several release-integrity defects that would have shipped a half-broken
tree — a fix reverted three times by stale-tree merges, and a secondary-smooth
penalty default that aborted an otherwise-valid fit — are repaired with
regression guards.

### Fitting / inference correctness
- **Multinomial per-class λ / EDF are rebuilt from the joint penalty (#561).** The
  outer REML/LAML loop now converges on genuine per-term smoothing rather than
  parking at its seed, so a multinomial fit recovers per-class structure instead
  of a fused, over-/under-smoothed surface.
- **`smooth_significance()` reference d.f. is floored at the joint null-space
  dimension (#1766)** so the likelihood-ratio p-value no longer collapses toward
  0 on a shrunk smooth, keeping the null false-positive rate calibrated.
- **The Marra & Wood null-space "double" penalty now defaults OFF for a secondary
  (scale / distributional) smooth (#1561)** — defaulting it on biased the
  location-scale fit toward homoscedasticity and collapsed the recovered
  log-sigma surface. `duchon` is excluded from the change (it carries no such
  penalty and its builder rejects the key), so a scale-block Duchon fit no longer
  aborts; an explicit user `double_penalty=` still wins.
- **`gam fit --family expectile` no longer aborts on the frailty guard (#1780).**
  The inner Gaussian-identity design carries no frailty; the CLI's default
  `frailty = Some(None)` is now cleared before the inner fit.
- **No spurious "Inferred gaussian-identity family" note when an explicit
  non-default `--family` is passed (#1781).**
- **A 1-D cyclic basis wraps on the data range instead of hard-erroring.**
- **High-EDF Gaussian observation intervals cover** — the residual-df scale keeps
  observation bands calibrated at high effective degrees of freedom (#1765).

### New public surface
- **`periodic=` on the radial builders `duchon()` / `tps()` / `thinplate()` /
  `matern()` (#580, #1778)** — a scalar or per-axis boolean, wired through Rust
  and the CLI, with validation that rejects bad axes, per-axis lengths, and
  non-positive or over-wide periods.
- **The expectile (LAWS) regression family is reachable from both the Python API
  and the CLI (#1777)** via `--family expectile` (inline-τ supported), routed
  through a shared dispatch seam so both interfaces agree.

### SAE / manifold
- **The SAE fit is routed through Rust FFI** with per-fit config overrides
  (separation-barrier strength / IBP-α), a `threshold_gate` rename (was
  `jumprelu`), out-of-sample v-projection, `atom_reconstruct`, and
  `coord_sparsity` (#1777); the SAE-manifold audit fixes real defects across the
  hybrid / routing / log-det / sparse paths (#1026).
- **Fisher steering state now round-trips through `save` / `load`,** so a reloaded
  model reproduces `steer()`'s dose instead of silently degrading to
  geometry-only.
- **A GPU device-resident exact encode kernel, sublinear massive-K routing, and
  jet / REML / arrow-Schur perf** — including a matrix-free reduced-Schur SLQ
  evidence log-det — all bit-identical to the CPU / scalar path.

### `gamfit` package (breaking)
- **Python-side numeric math that violated the thin-wrapper SPEC has been removed;
  the capabilities that exist in Rust are reached through the FFI.** Ported to
  Rust (behavior preserved): `partial_dependence`, `variance_share`, the sparse-
  and linear-dictionary out-of-sample `transform`, and the cyclic difference
  penalty. Removed with no replacement (breaking):
  `gamfit.align` (Procrustes alignment), the `sae_benchmark` /
  `sweep_sae_benchmark` / `format_sae_benchmark_markdown` harness,
  `activation_statistics`, `recommend_sae_hyperparams`, the EV-vs-K frontier
  research helpers (`sae_ev_vs_k_frontier` / `ev_knee_k`), and
  `Model.posterior_predictive_check`.
- **`coordinate_range` / `typical_shape` are consolidated into
  `shape_uncertainty`** (which returns the analytic per-atom band as
  `coords` / `mean` / `sd` / `lower` / `upper`).
- **ALR simplex fits are no longer auto-whitened to Aitchison geometry** — no Rust
  FFI exists for the whitener, so an ALR fit again depends on the (arbitrary)
  reference component. Use the default CLR (or ILR) representation, which is
  already Aitchison-isometric, for a reference-free simplex fit (#1549
  auto-whitening removed, per SPEC).

### Build / release integrity
- Restored the #1780 expectile-frailty fix after stale-tree merges reverted it,
  and pinned it with a CLI regression test so a future clobber fails loudly.
- Added a regression guard that a scale-block `duchon()` smooth fits under the
  #1561 default-off, and that an explicit `double_penalty=` on Duchon is still
  rejected.
- Cleared build-gate violations that would have failed the wheel build mid-flight
  (a banned `debug_assert!` guarding an `unsafe` load promoted to an always-on
  `assert!`; stray diagnostic scaffolding; a test-only dispersion oracle moved
  into `#[cfg(test)]` scope; a `sae_manifold_fit` arity mismatch at an internal
  IBP call site), plus build.sh usability and OOM-recovery hardening.

Everything reachable through the existing API stays backward-compatible except
the explicitly-listed `gamfit` removals.
## v0.3.138 — gam 0.3.138 / gamfit 0.1.240 (2026-06-30)

crates.io + PyPI release of the post-0.3.137 correctness wave. A cluster of
silent-no-op and frame-consistency bugs are fixed at the root — each with a
regression test — the monotone shape-constrained REML fix is completed for its
binding-constraint face, several GPU paths are proven bit-identical to CPU, the
Gaussian P-spline/thin-plate objective gains fast paths, and two latent
build-breakers (an unused import and eight hygiene-gate violations that would
have failed the wheel build mid-flight) are cleared. Everything reachable
through the existing API stays backward-compatible.

### Fitting / inference correctness
- **Tensor smooths honor per-margin periodicity and `bs=c(...)`.** A `te(...)`
  mixing periodic and non-periodic marginals (or per-margin basis overrides via
  `bs=c(...)`) built every marginal from the first margin's spec; each marginal
  now carries its own periodicity and basis kind (#1751, #1752).
- **`smooth_significance()` LR p-value no longer collapses to ~0.** When a smooth
  shrinks onto its linear null space the reference degrees of freedom could fall
  to zero, collapsing the likelihood-ratio p-value; the reference d.f. is now
  floored at the term's null-space dimension (≥ its EDF), keeping the null
  false-positive rate calibrated (#1766).
- **`survival_likelihood=` is rejected on a non-`Surv()` response (#1767).** A
  request like `fit(data, "time ~ s(x)", survival_likelihood="weibull")` used to
  drop the knob silently and fit an ordinary Gaussian GAM on the raw event-time
  column. It now fails loud at materialization, symmetric to the `family=`
  survival guard and the survival-only formula-term guard (#371).
- **Monotone shape-constrained smooth — binding-constraint face (#509).** The
  outer REML/LAML analytic gradient is now frame-consistent with the cost even
  when the monotonicity constraint binds at the inner optimum, so a monotone fit
  on already-monotone data no longer parks at its under-smoothed seed. Completes
  the #509 fix begun in 0.3.137.
- **`ard_per_atom` is wired to the native ARD prior (#240).** The SAE flag was a
  silent no-op (a registry penalty deliberately skipped on every SAE path); it
  now toggles `native_ard_enabled`, observable in the born-atom count and fitted
  coordinates.
- **Multinomial per-class λ/EDF rebuilt from the joint penalty (#561); endpoint
  `bc=clamped` interior-quality guarded and tensor endpoint BCs rejected (#500);
  per-atom `log_lambda_smooth` grows when a structure move grows K (#357);
  irrelevant double-penalty smooths select out via per-term shrink (#1266); the
  distilled-encoder honesty probe is cold-started (#1166); the objective-grid
  per-axis seed refinement reaches asymmetric corners.**

### SAE
- **Closed-form fast paths honor `random_state` (#178)** via a deterministic LCG
  mirrored between Python and Rust.
- **Scale-invariant isometry Gauss–Newton curvature; the gauge is re-enabled
  (#795). Log-det θ-adjoint cross-row Woodbury off-diagonal corrected
  (#1625/#1416); barrier strength derived from REML evidence (#1610);
  device-resident Direct-solve engagement with symmetric Schur fixtures
  (#1017/#1551).**

### Survival
- **Predict-query times accept negative and `+inf` (#965)** as boundary
  evaluations instead of raising.

### Performance
- **Gaussian P-spline / thin-plate fit perf-core (#1689):** REML-objective fast
  paths with a profile-test guard.
- **Sphere (S²) GPU dtoh path:** pinned + parallel transpose + buffer pool
  (#1709); the 40M-element transpose in `to_host_array` removed (GPU
  6.6s→0.27s, #1741); decomposition GEMM routed through `fast_ab` (#1738).
- **`trace_product_sparse` rayon parallelization restored (#759); gauge identity
  section short-circuited in `restrict_design`/penalty (#1737); dense-GEMM device
  dispatch registered and made transpose-free (#1735); Matérn/GP κ-loop sped up
  with the verbosity flag (`gam._rust.set_log_level`) re-exposed (#1688); wiggle
  separation bound + batched REML gradient (#1607).**

### GPU parity / robustness
- SAE reconstruction CPU↔GPU bit-identity via row-jet K-scale + sparse-route
  (#1026); honest fail-loud routing (#1209); proven GPU↔CPU parity on V100 with
  principled bands + fmad sweep (#415, #1175); CI-fast parity, fail-loud oracle,
  and false-routing guard (#1412, #988); arrow-Schur NVRTC arch-pin and
  deficit-aware Gershgorin ridge bump for non-PD rows.

### Build / test hardening
- **Un-RED the workspace build.** Removed an unused `ShapeBuilder` import in
  `gam-terms` (a `warnings = "deny"` hard error left by the #1709 sphere-GPU
  work) and cleared eight `build.rs` ban-scanner violations across
  `gam-solve`/`gam-models`/`tests` (a `let _` discard, two `#[ignore]`d GPU
  diagnostics, a `GAM_REQUIRE_GPU` env read, a mis-named `#[cfg(test)]` module,
  and a three-`assert!(true)` scratch test file). Both gates would have failed
  the wheel build ~12 min in; they had slipped through CI because the heavy
  build job self-throttles and kept getting skipped under the steady commit
  stream.
- Concurrency guards (`OnceLock`-in-rayon #1253, `nested_prefix` dispatch #1254)
  made behavioral and workspace-wide; `scale_dimensions` anisotropy validated for
  `thinplate()` (#1676); numerous orphaned smooth / Matérn / SO(3) / SAS / BMS
  test guards re-homed and bound to production penalties (#1601, #1274, #1629,
  #855, #388, #370, #1260, #1261, #1246, #1255, #1091); Test-shard CI timeout
  raised to the 6h runner max.

## v0.3.137 — gam 0.3.137 / gamfit 0.1.239 (2026-06-30)

crates.io + PyPI release of the post-0.3.136 correctness wave. Shape-constrained
REML, adaptive-ψ custom families, and three user-facing API contracts are fixed
at the root, each with a regression test; one inner-loop hot path is made
n-free. Everything reachable through the existing API stays
backward-compatible.

### REML / fitting correctness
- **#509 — monotone shape-constrained smooth no longer over-smooths.** The outer
  REML for a `shape=monotone_increasing` (box-reparam) smooth projected the
  penalty roots in the ORIGINAL (pre-`Qs`) frame while the Hessian half of the
  LAML pair lived in the TRANSFORMED (post box-reparam `T`, post-`Qs`) frame, so
  under the non-orthogonal cumulative-sum reparameterization `ZᵀSZ` mixed two
  coordinate systems, the analytic outer gradient disagreed with finite
  differences, the trust region rejected every step, and the fit parked at its
  under-smoothed seed (mono RMSE > free RMSE on already-monotone data). Both
  halves now live in one transformed frame (the no-op identity when `Qs = I`), so
  the non-binding monotone fit recovers the truth as well as the unconstrained
  fit. Also hardens the binding-constraint regression and removes a
  parallel-test temp-CSV race.
- **#901 — adaptive-ψ custom families: data-only Jeffreys information + ψ-gated
  Firth.** The projected-logdet REML gradient now matches finite differences for
  spatial-adaptive-hyper custom families, with a 659-line FD agreement suite.

### Contrast / compositional / comparison API
- **`Model.difference_smooth` sign corrected.** A pair `(level_1, level_2)` now
  returns `ŝ(level_1) − ŝ(level_2)` (the mgcv `plot_diff` convention); the design
  difference was previously formed as `design(level_2) − design(level_1)`, so the
  reported contrast was the exact negation of its `level_1`/`level_2` row labels.
  The confidence band (a quadratic form) is unchanged.
- **`gamfit.clr` / `alr` / `closure` accept a single 1-D composition.** The
  natural call `clr([0.2, 0.3, 0.5])` raised an opaque
  `TypeError: 'ndarray' object is not an instance of 'ndarray'` because the FFI
  only accepted a 2-D `(rows, parts)` array; a 1-D composition is now promoted to
  a single row and returned as 1-D coordinates matching the batch row.
- **`compare_models` refuses cross-`n` comparisons.** AIC / REML-LAML evidence
  scales with the observation count, so comparing two same-family fits on
  different-sized data (e.g. n=500 vs n=100) used to declare the smaller-`n` model
  the winner by a Bayes factor ~1e14–1e18. It now fails loud on mismatched `n`,
  mirroring the existing different-family guard; fits that do not record `n`
  (legacy / O(n) scan payloads) stay unconstrained.

### Performance
- **#1033 — n-free κ-trial lane.** The ALO leverage-barrier stabilizer (an
  outer-optimizer aid, never part of the REML/LAML criterion) is skipped on the
  ψ-keyed sufficient-statistic cache lane whose realized rows are frozen at the
  pinning ψ, removing an O(n·k) hat-value pass per in-window κ-trial without
  changing any fitted result.

### Test hardening
- #1260 binds the equivariant-atom bandwidth gate to the shipped penalty
  (replacing a vacuous self-objective test); #1261 restores the oversmoothed-λ
  regime for the average-derivative one-step gate after penalty renormalization;
  the joint-Newton weak-band mode is placed strictly inside the rank band; #855
  restores the tight SAS dβ/dε observed-Jacobian FD guard.

## v0.3.136 — gam 0.3.136 / gamfit 0.1.238 (2026-06-30)

crates.io + PyPI release of the post-0.3.135 correctness wave. A cluster of
smoothing/basis, REML-convergence, structure-search, geometry, survival and
inference bugs are fixed at the root, each with a regression test. Everything
reachable through the existing API stays backward-compatible; two changes adjust
default basis sizing toward mgcv (a wigglier fit is still one explicit `k=`
away).

### Smoothing / basis fixes
- **#1680 — the default univariate smooth basis is capped mgcv-like.**
  `heuristic_knots_for_column` grew the default B-spline basis with `n` (20
  internal knots / a 24-function cubic basis for any column with ≥80 unique
  values; the `n^{1/3}` ceiling only engaged above ~8000 unique values, so it was
  dead in practice). That over-rich default over-parameterized weak-signal
  additive fits and let the outer REML optimizer leak truth into surplus columns
  the penalty could not shrink (truth-RMSE ≈0.39 vs mgcv's ≈0.09 on a
  near-collinear 4-smooth n=120 fit). The default is now a flat 8 internal knots
  (basis dim ≈12, close to mgcv's univariate `k=10`); columns with ≤32 unique
  values keep their previous knot count exactly, and an explicit `k=` always
  wins. Same defect class as the thin-plate over-sizing in #1074.
- **#1731 — Matérn realized basis now grows with requested `k`.** The auto length
  scale was seeded `k`-blind (`max_range/√n`), so once the requested centers
  packed denser than that fixed scale could resolve, neighbouring radial bases
  went numerically collinear and the #755 rank-reduce guard dropped them — the
  basis saturated and even *decreased* for large `k` (`k=150 → 104` realized).
  The auto length scale is now density-adaptive (`max_range/√max(n,k)`, the same
  fill-distance law the Duchon promotion uses): bit-identical to the old seed
  whenever `n ≥ k`, and shrinking with `k` past that so the centers stay
  independent (`k=150 → 150`). Only the auto sentinel is touched; an explicit
  length scale is never overridden and the rank-reduce guard remains the
  last-resort degenerate-data net.

### REML / optimizer
- **#1033 — the κ/ψ smoothing window is n-invariant at BOTH edges.** The κ line
  search could overshoot ABOVE the maximal-rank band to a ψ where the conditioned
  Gram drops rank, soundly refusing the n-free design-realization skip and
  tripping two O(n) `reset_surface` passes (the n=16000 fast-ladder regression).
  A symmetric `rank_stable_psi_ceiling` (the twin of the existing low-edge floor)
  now clamps the optimizer's ψ upper bound to the top of the maximal-rank band —
  a pure O(nodes·k³) k-space property, inherently n-independent. The κ-optimum
  lives inside the band, so the clamp only excludes over-fit length scales.
- **#1690 — a Gamma flat-valley REML stall is no longer mis-reported as
  non-converged.** A single-smooth `s(x,k=12)` n=600 Gamma+log fit reaches the
  genuine optimum but the in-loop cost-stall guard sampled a warm-start-sensitive
  ρ-gradient just above its score-relative bound and halted non-converged, which
  in turn triggered wasted deterministic-replay ARC retries (the actionable slice
  of "Gamma ~7× slower than mgcv at equal accuracy"). `outer_converged` is now
  reconciled against the authoritative gradient of the fit actually shipped, gated
  strictly on the flat-valley stop reason; a genuinely non-stationary floor (and
  the #1426 stuck overfit, |g|≈11 ≫ bound) still reports non-converged.

### SAE structure search
- **#1556 — birth/fission no longer panics.** Structure-search grow moves bumped
  the dictionary size and `ρ.log_ard` but left `ρ.log_lambda_smooth` at the old
  length, so the next `assemble_arrow_schur` indexed out of bounds. Both grow
  paths now push an inherited per-atom smoothness strength (born inherits atom 0;
  a fissioned child inherits its parent), and `assemble_arrow_schur` validates the
  length so any future grow path that forgets surfaces a clear `Err`.
- **#977/#1026 — the born-atom topology race is scored by proper REML.** The birth
  race scored each candidate basis with a hand-rolled `½·SSE + ½·log|H|` Laplace
  term at a stamped `λ=1` on the raw curvature Gram — not commensurable across
  bases, so a periodic basis's `(2π)⁴` curvature energy lost a perfect circle to a
  straight line (and a cylinder to a sphere). Candidates are now scored by a
  rank-aware REML/LAML with an estimated λ̂, so the heterogeneous-dictionary races
  pick the topology the evidence supports.

### Geometry / linear algebra
- **#1641 — IBP θ-adjoint cross-row Woodbury logdet channel corrected.** The
  cross-row Woodbury pass in `logdet_theta_adjoint` carried a spurious ½ (a
  ρ-trace convention) while differentiating the full `log|H|`, dropped the factor
  of 2 on the symmetric u-changing term, and double-counted the `i=j` self
  curvature already handled by the diagonal channels. The pass now restricts to
  the `i≠j` off-diagonal with full-trace coefficients, mirroring the known-good
  #1416 ρ-trace cross-row pass and matching the dense finite-difference oracle.

### Survival
- **#1717 — `survival_at(t|x)` is invariant to the placeholder time column.** The
  default 64-point survival grid floored its upper edge to the training support
  but let a large placeholder exit stretch it past, coarsening every in-range cell
  and drifting the interpolation off the true curve (up to ~14%). When the fitted
  model carries a training-time upper bound the grid now spans exactly that
  support; query times beyond it are handled by extrapolation (#1595), not by the
  grid. The dual of #896.

### Inference
- **#1722 — Beta posterior credible intervals are no longer ~4-5× too narrow.**
  `laplace_gaussian_fallback` rescaled draws by `dispersion().sqrt_phi()`, but
  Beta's IRLS working weight already folds φ into the stored penalized Hessian, so
  `Vb = H⁻¹` needs no extra dispersion factor. The per-draw scale is now the
  coefficient-covariance scale `summary()`'s Wald SE is built from — `σ̂²` for a
  profiled Gaussian (a no-op for Gaussian/location-scale/survival) and `1.0` for
  Beta and the other fixed-scale families, fixing only Beta.

### Internal / CI
- The `[profile.test]` base now optimizes the workspace numerical crates
  (`opt-level = 2`), not just dependencies, so the heaviest solver-bound tests
  (#979 survival, #1593 competing-risks) finish inside nextest's 600s per-test
  cap instead of dying as opaque SIGKILL timeouts. Numerically identical.
- `build.sh`'s inner timeout is overridable via `GAM_BUILD_TIMEOUT`.

## v0.3.135 — gam 0.3.135 / gamfit 0.1.237 (2026-06-30)

crates.io + PyPI release of the post-0.3.134 correctness-and-performance wave.
A broad set of smoothing/REML, survival, geometry and inference bugs are fixed
at the root, several hot paths are made meaningfully faster, and the `gamfit`
Python surface gains additive SAE keyword aliases. Everything reachable through
the existing API stays backward-compatible.

### Smoothing / REML fixes
- **#1654 — convex/concave shape smooths no longer park in the linear corner.**
  The double-penalty nullspace ridge under the order-2 box reparameterization
  was rebuilt from scratch instead of transformed by the same congruence
  `S ↦ TᵀST` as the wiggliness penalty, decoupling the level/slope and
  wiggliness scales and driving curvature-constrained fits to a near-straight
  line (EDF ≈ 1.5) for a seed/`k`-specific subset. The exact congruence is
  restored for the curvature ridge; the monotone path keeps its #509 projector.
- **#509 — monotone REML λ-search no longer parks at the integer seed.** The
  cost-stall guard keyed its keep-descending escape on a fixed absolute gradient
  ceiling, so a shape-constrained inner solve with a non-binding constraint
  stalled near the seed while the projected gradient still descended strongly
  (over-smoothing already-monotone data). The escape is now scaled to the score.
- **#1629 — Matérn smooths no longer over-smooth 2-D surfaces.** Matérn now
  routes through the same length-scale auto-init sentinel as thin-plate so the
  basis seeds the resolving regime instead of a degenerate global scale.
- **#1676 — `scale_dimensions=True` now engages anisotropy for thin-plate.** A
  multi-axis `tp` term is rewritten to its mathematically-equivalent anisotropic
  s=0 Duchon twin (the thin-plate kernel `r^{2m−d}` is the s=0 Duchon kernel), so
  per-axis tension ARD engages exactly as for `duchon()`/`matern()` instead of
  the flag being a silent no-op. Default (flag off) and 1-D `tp` are unchanged.
- **#1269 — thin-plate basis is exactly translation-invariant.** The strict
  basis-conditioning gate is split out and pinned at the bit level.
- **#1476 — double-penalty no longer over-shrinks a supported smooth.** A budget-
  exhaustion best-feasible substitution used a bare early `return` that bypassed
  the multi-start keep-best loop and could ship a degenerate box corner; it now
  flows through keep-best as a non-converged candidate.
- **#1033 — the κ/ψ smoothing window is now n-invariant.** Even-spaced capped
  diameter sampling and a rank-stable ψ floor anchored at the optimizer seed kill
  an n-dependent shift in the outer optimizer's box.
- **#901 — iso-κ joint-REML outer-gradient FD oracles re-homed and verified.**

### Survival fixes
- **#965 — survival FFI rejects negative times; `S(0)=1`.** Negative/NaN/Inf
  times are rejected at the Python→Rust boundary and the parametric fallback
  guards the `exp(-∞·0)=NaN` origin case.
- **#1595 — survival/cumulative-hazard extrapolation policy threaded into the
  dense Rust FFI kernels**, so `S=exp(-H)` holds past the grid in both the
  chunked and CSV paths.
- **#392/#369 — fit-to-completion guards restored for non-linear survival
  baselines** (real convergence asserted across all baseline targets).

### Inference / families fixes
- **#332 — near-constant Gaussian response is rejected with a clean error**
  instead of producing a degenerate fit.
- **#1655 — the GPD tail estimator accepts light (k<0) tails** (σ from the
  un-shrunk k, matching ArviZ/loo) instead of returning `None`.
- **#1621 — debiased point/contrast prediction handles inert categorical
  bookkeeping columns** via lenient encoding for non-required columns.
- **#1101 — the multinomial per-class probability-SE calibration test** is
  replaced with a valid over-refit calibration (the prior test was statistically
  degenerate).
- **#1561 — the final location-scale β̂ refit at ρ\* is seeded warm from the
  outer optimum** instead of cold, fixing a basin-fragility KKT cert-refusal
  crash on stiff two-block fits.

### Geometry
- **#1637 — genuine Stiefel canonical-metric logarithm for k≥2**, anchored to
  Y⊥ to kill spurious π rotations, with an exhaustive `Log∘Exp=id` sweep and a
  square-input completion guard.
- **#1661 — CLR `simplex_exp_map` rejects a non-finite tangent** with an error
  instead of returning `Ok(NaN)`.

### Python (`gamfit`)
- **#159/#160/#178 — additive SAE keyword aliases**: `assignment` /
  `assignment_prior`, `K` / `n_atoms`, and `random_state` wiring, resolved
  end-to-end into the Rust FFI with eager conflict detection.

### Performance (value-preserving)
- **#1575** cuts the Firth/Jeffreys outer-Hessian cost on binomial/logit REML.
- **#759** restores the rayon parallel reduction in `trace_product_sparse`.
- **#1082** brings the competing-risks CIF quality case from 439 s to 122 s by
  not expanding the ρ₀ offset on an untagged inner-failure pre-warm.
- **reml Jeffreys drift** GEMM-izes the H_Φ curvature-drift contraction (was a
  bounds-checked scalar triple loop dominating competing-risks Weibull fits).
- **#1033 / #979** make the ψ-gram a true sufficient-statistic reduction and
  bound the marginal-slope continuation pre-warm.

### Known-limitation note
The GAMLSS location-scale engine/reference parity and inner-solve convergence
cluster (#1607) remains under active work; those `gam-models` tests are not yet
green and no user-facing API depends on them.
## v0.3.134 — gam 0.3.134 / gamfit 0.1.236 (2026-06-29)

crates.io + PyPI release of the post-0.3.133 correctness-and-performance wave:
two user-visible inference/prediction bugs are fixed, the multinomial save model
gains a first-class per-penalty EDF channel, and the Firth/Jeffreys REML and SAE
log-det hot paths are substantially faster while staying numerically identical.
The `gamfit` Python API surface is additive only — multinomial model metadata now
carries `edf_per_penalty`; everything else reachable through the existing API is
unchanged.

**debiased_functional restored for parametric-term Gaussian models (#1622)**
- `debiased_functional` no longer aborts with "model does not carry the weighted
  Gram X'WX" on every Gaussian model that has a parametric (non-intercept) term
  (`y ~ x`, `y ~ s(x) + z`). Under column-conditioning the weighted Gram `X'WX`
  is a genuine congruence object — it transforms by exactly the same map as the
  penalized Hessian — so it is now back-transformed into the original basis
  rather than unconditionally dropped, letting the debiased-functional Riesz
  engine recover `S(λ)·β`. The same fix restores the exact WPS corrected-EDF term
  `tr(X'WX·Σ_ρ)` (congruence-invariant, so it matches the internal-basis value
  bit-for-bit) for the whole `y ~ x` / `y ~ s(x) + z` class.

**Point/contrast prediction under the full training schema (#1621)**
- The `x0` design for point and contrast predictions is now built under the full
  training schema, so predictions no longer mis-align when the prediction frame
  carries fewer terms than the fitted model.

**Multinomial per-penalty EDF is now first-class (#1219, #715)**
- `MultinomialSavedModel` gains an `edf_per_penalty` field (one entry per
  smoothing parameter, `rank(S_k) − λ_k·tr(H⁻¹ S_k)` clamped to `[0, rank]`),
  surfaced through the Python multinomial metadata. Previously the per-class
  `edf_per_class` field was overloaded to also answer per-penalty collapse
  detection; with double-penalty smooths the two vectors have different lengths,
  so one consumer always read a wrong-length vector. Both quantities are now
  independently correct.

**Firth/Jeffreys REML performance (#1575)**
- Binomial/logit REML with default-on Firth/Jeffreys bias reduction no longer
  rebuilds the entire `FirthDenseOperator` (the O(n·p²) design Gram, the O(p³)
  identifiable-subspace eigendecomposition, the design clones) on every inner
  Newton iteration. The β-independent design factor is built once per PIRLS solve
  and memoized; only the per-η reduced core is rebuilt. The converged β/λ/EDF/score
  are bit-for-bit unchanged, pinned by an operator-equivalence oracle.

**SAE log-det trace performance (#932)**
- The SAE reconstruction log-det / α-trace channels are back on the hand
  closed-form `row_jets_for_logdet` (a measured 25–57× throughput win over the
  Taylor-jet cutover, bit-identical to ≤1.4e-15), with row-local Takahashi
  selected-inverse fast paths layered on top. The Taylor jet is retained as a
  `#[cfg(test)]` correctness oracle, not deleted.

**BMS Firth/Jeffreys outer-gradient correctness (#1607)**
- The explicit Firth/Jeffreys value ψ-derivative is now carried on the BMS
  batched outer-gradient path, so its `objective_theta` matches the hypercoord
  gradient (and the centered FD of the Firth-corrected outer value) for Jeffreys
  BMS spatial fits.

**Build / test hygiene**
- A wave of test-infrastructure hardening across gam-math, gam-linalg, gam-sae,
  and the multinomial/dispersion oracles (visible assertions, oracle relocation,
  removal of `let _` / ignored-bench laundering), and the SAE row-jet oracle
  fixtures are lifted into the PD basin (#1625) so they converge and assert.

## v0.3.133 — gam 0.3.133 / gamfit 0.1.235 (2026-06-29)

crates.io + PyPI release of the SAE reconstruction-fidelity, penalty-spectrum
robustness, and Firth-performance wave landed since gam 0.3.132 / gamfit 0.1.234.
The headline changes stop the SAE hybrid collapse from over-simplifying a
genuinely curved atom, remove a spurious P-IRLS rejection on high-rank smooths at
extreme λ, and cut the dominant cost of the Firth/Jeffreys outer Hessian. The
`gamfit` Python API surface is unchanged — these correct and accelerate the
behavior reachable through the existing API.

**SAE hybrid-collapse EV preservation (#1610, #1026)**
- A curveable `d = 1` atom that is doing real reconstruction work is no longer
  collapsed to its straight linear tail. The hybrid-split selector now gates each
  collapse on the reconstruction explained variance it would cost, vetoing a
  collapse that would raise the full reconstruction SSR by more than 1e-3 of the
  target's total centered variance (the observed 1.0 → 0.748 EV over-collapse on
  small, low-amplitude fixtures). Lossless / EV-neutral collapses still collapse
  freely. As part of the same fix a degenerate all-stationary atom image now
  reports zero total turning (a point has no arc to turn through) instead of
  refusing, while a partial cusp still refuses — both pinned by new geometry
  regression tests, and the integration test now asserts the EV-axis
  discrimination the gate actually performs.

**Penalty-spectrum PSD floor (#1619)**
- A high-rank thin-plate / Duchon penalty (p ≈ 200) assembled and reparameterized
  at extreme λ during REML no longer fails the inner P-IRLS solve on roundoff: the
  strict PSD classifier now accepts a negative eigenvalue up to the larger of the
  machine-ε floor and a relative numerically-PSD floor (1e-8·scale ≈ √ε), snapping
  it to zero. Genuine indefiniteness is O(1) relative and is still rejected far
  above either floor.

**Firth/Jeffreys REML performance (#1575)**
- The default-on Firth/Jeffreys outer REML Hessian for binomial/logit is
  substantially faster: the single-index sub-blocks of the exact Tierney-Kadane
  mixed second directional derivative — previously rebuilt for every one of the
  k(k+1)/2 penalty pairs against the same identity right-hand side — are now
  precomputed once per direction and reused across all pairs (~37% fewer of the
  dominant O(n·r²·p) applies at k = 6). The converged β/λ/EDF/score are unchanged,
  pinned bit-for-bit by a cached-vs-per-pair oracle test.

**Gaussian weighted prior-weight semantics (#1617, #1618)**
- The intended Gaussian `weights` convention is locked down by a contract test:
  for a fixed-dispersion family (Poisson) a weighted fit reproduces the
  row-expanded fit exactly, while a Gaussian identity-link fit treats `weights` as
  prior (inverse-variance) weights — rescale-invariant under w → c·w and *not*
  equivalent to row replication (its profiled scale divides by the row count,
  matching mgcv / Wood 2017 §6.2.7). Net behavior is unchanged; this rebuts the
  #1617/#1618 false-premise reports and prevents silent drift.

## v0.3.132 — gam 0.3.132 / gamfit 0.1.234 (2026-06-29)

crates.io + PyPI release of the non-Gaussian-family, survival-identifiability,
and convergence-hardening wave landed since gam 0.3.131 / gamfit 0.1.233. The
headline fixes make GP/interval-coefficient smooths fittable under the
exponential-family GLMs that previously rejected them, stop the survival
marginal-slope hang at its root, and remove the Tweedie smoothing boundary bias.
The `gamfit` Python API surface is unchanged — these broaden and correct the
behavior reachable through the existing API.

**Non-Gaussian standard-family terms (#1615, #1616)**
- `matern(x, z)` GP smooths and `bounded(x, min, max)` interval-coefficient
  terms now fit under Poisson, Gamma, Negative-Binomial, and Tweedie families.
  The standard-family observation builder previously had only Gaussian/Binomial
  arms and hard-aborted ("not supported for …") once the coefficient search
  reached any other family; it now derives the score, Fisher weight, observed
  η-Hessian, and its η-derivative analytically for each. (Beta remains
  deferred — its 3rd μ-derivative needs the tetragamma — and still bails.)

**Survival marginal-slope identifiability (#979)**
- Survival `marginal-slope` fits that shared a spatial basis between the
  marginal and log-slope channels could run to the outer wall-clock timeout: the
  full row-Hessian compile attributed the entire shared surface to the
  log-slope block and collapsed it, leaving a quadratically-flat near-null
  direction the inner joint-Newton could never certify. A W-orthogonal *partial*
  reduced-log-slope reparam now drops only the marginal-explained log-slope
  directions and keeps the survivors, so the joint penalised Hessian is
  full-rank by construction and the outer deadline is demoted to a pure
  backstop. (This release also adds direct unit coverage of the new reparam and
  silences a false "pilot-curvature trap" warning it emitted on every success.)

**Tweedie smoothing (#1477)**
- Tweedie `s(x)` P-spline fits no longer ship a right-boundary blow-up /
  EDF-inflated biased mean: the dispersion φ is now held fixed across the
  smoothing-parameter search (matching mgcv and the existing Negative-Binomial θ
  handling), making the REML criterion stationary. Reported dispersion and
  standard errors are unchanged — φ is still refreshed at the final fit.

**GAMLSS Gaussian location-scale (#1561)**
- Gaussian location-scale (and its wiggle variant) now select the flexible
  low-λ log-σ basin instead of over-smoothing the scale predictor; the deeper
  seed screening can no longer discard the heteroscedastic fit via the
  over-smoothed seed's flat-Fisher "looks-cheap-early" proxy. The keep-best
  rule is unchanged (lowest-cost), so the result is provably non-worsening.

**Cyclic smooths (#1593)**
- A cyclic/periodic smooth's fitted curve is now invariant to the arbitrary
  `period_start` phase origin (worst drift ~2e-2 of signal range → ~1e-10): the
  uniform cyclic knot grid is anchored to a canonical lattice rather than
  rigidly to the user's seam, and the dense and streaming evaluators wrap data
  into the same shifted window so fit- and predict-time designs agree.

**SAE manifold penalties (#1610, #1026)**
- The SAE separation-barrier strength `μ_C` is now data-derived from dictionary
  overcompleteness (`K / reachable_rank`) and dimensionless, so it is invariant
  under a global corpus rescaling instead of the hand-picked constant `10.0`;
  the decoder-repulsion strength tracks it as a fixed fraction.

**Firth/Jeffreys REML performance (#1575)**
- Binomial/logit REML with Firth/Jeffreys bias reduction is substantially
  faster: the β-independent design factor (Gram, identifiable-subspace
  eigendecomposition, reduced design) is built once per inner P-IRLS solve
  instead of every Newton iteration, and the seed-screening prepass uses a
  coarse inner tolerance. Converged fit results are unchanged (now pinned by a
  direct factored-vs-full-operator equivalence test).

**Regression guards**
- Reference-free guards added for competing-risks CIF invariance to cause-label
  permutation (#1593, Rust + Python) and 2-D thin-plate truth recovery (#1074),
  so those properties stay protected on CI nodes without mgcv/R.

## v0.3.131 — gam 0.3.131 / gamfit 0.1.233 (2026-06-29)

crates.io + PyPI release of the generation-contract, structured-additive-model
(SAE) collapse-prevention, and Firth/Jeffreys outer-LAML wave landed since gam
0.3.130 / gamfit 0.1.232. The headline fix completes the response-scale
`generate` contract for conditional transformation-normal (CTM) models; the rest
hardens REML/LAML convergence and the SAE penalty stack. The `gamfit` Python API
surface is unchanged.

**Conditional transformation-normal models (#1613)**
- `gam generate` on a CTM model drew synthetic responses on the *latent* N(0,1)
  scale: it required the outcome column `y`, and the per-row mean of the draws
  moved the *wrong way* with the covariate. It now draws genuine response-scale
  `Y` by inverting each row's monotone transform on a standard-normal quadrature,
  so the draws track `E[Y|x]` (verified: a fit to `E[Y|x] = 2 + 0.9x` centers its
  `x = −1, 0, 1` draws on the increasing sequence, no `y` column required). This
  completes the predict/generate response-scale contract begun in #1612.

**Firth/Jeffreys outer LAML (#1607)**
- For a ψ hyperparameter that reshapes the design (Matérn/Duchon length-scale),
  the joint-Jeffreys penalty `Φ = ½ log|Zᵀ H Z|₊` depends on ψ explicitly, but the
  BMS batched outer gradient dropped both the value term `−∂_ψΦ` and its β-coupling
  `−∂_β∂_ψΦ`. Both are now folded into the outer LAML gradient and Hessian, gated
  on `joint_jeffreys_term_required()` so well-conditioned/non-Jeffreys fits stay
  byte-identical.
- The explicit-parameter ψψ second derivative now carries the full
  conditioning-gate curvature `G''·U` (not just the gate motion `G'`), making it
  the exact second derivative of the gated value and consistent with its own
  gate-aware gradient inside the gate's transition band.
- A homoscedastic (flat) scale ridge could exhaust the coupled joint-Newton budget
  while genuinely converged on the identifiable subspace; the convergence
  certificate now also admits the objective-plateau precondition
  (`Δobj ≤ objective_tol`) alongside the residual-stall window, both still gating
  the rigorous Newton-decrement bound.

**SAE manifold penalties (#1610, #1017)**
- Decoder-repulsion collapse-prevention strength is now *derived* from the
  separation-barrier strength and energy-normalized, so it is scale-invariant
  rather than a hand-tuned absolute constant; the separation-barrier collapse
  norm-floor is likewise data-relative (to `maxₖ‖Bₖ‖²`) instead of an absolute
  magic number. Collapse-prevention curvature is engaged on the matrix-free/framed
  production path.
- The co-collapse acceptance bar is now calibrated against the dictionary's
  *reachable* geometric rank `Σₖ rank(Φₖ)` (read from each chart design alone, not
  the decoder magnitude) instead of the nominal coefficient count `Σₖ basis_sizeₖ`,
  which over-stated what a curved nonlinear dictionary can span and biased the
  linear PCA ceiling high. The bar can only move down from the old value.
- (GPU) The SAE G-matvec output accumulation now uses `atomicAdd`, fixing a data
  race when multiple blocks accumulate into the same output element for
  co-occurring row atoms.

**Other fixes**
- #1074: the Gaussian over-smoothing seed safety-net now extends past the
  screening cap, curing weak-signal spatial over-fit.
- #979: a *measured* KKT-refusal gate for the survival marginal-slope phantom null
  direction — projection engages only when a near-null direction is a measured
  phantom (zero gradient residual), never when it is genuinely driven, demoting the
  wall-clock deadline from load-bearing to a backstop.
- #1605: the `sz` factor smooth is exempted from owner-residualization.

**Performance (#1575)**
- Multi-slot outer-eval LRU reuses revisited ρ-points across the REML outer loop;
  redundant per-penalty Firth directions are hoisted out of the O(k²) TK
  outer-Hessian loop; SIMD 4-row batch kernels for the binomial location-scale
  directional derivatives.

## v0.3.130 — gam 0.3.130 / gamfit 0.1.232 (2026-06-28)

crates.io + PyPI release of the prediction/generation-contract and
GAMLSS-convergence wave landed since gam 0.3.129 / gamfit 0.1.231. The headline
fixes repair the response-scale `predict`/`generate` contract for conditional
transformation-normal (CTM) models and for Beta regression, and replace observed
with expected (Fisher) information in the negative-binomial and binomial
location-scale curvature so those GAMLSS fits converge on well-posed data. The
`gamfit` Python API surface is unchanged.

**Conditional transformation-normal models (#1612)**
- `gam predict` returned the probability-integral transform `h(y|x)` of the
  *supplied* response as both `linear_predictor` and `mean`, which wrongly
  required the outcome column at predict time and made the reported "mean" sweep
  with `y` at fixed `x`. It now returns the genuine response-scale conditional
  mean `E[Y|x] = E_{Z~N(0,1)}[h⁻¹(Z|x)]` — a function of the covariates alone —
  computed by inverting the monotone transform on a standard-normal quadrature
  (midpoint rule in probability space, shared fine `y`-grid I-spline inversion).
  A covariate-only frame now predicts without `y`.
- `gam generate` shared the same broken plug-in path and drew `N(h(y|x), sd)` on
  the latent scale, so synthetic responses moved the *wrong way* with the
  covariate. It now draws response-scale `Y` tracking `E[Y|x]` (verified: draws
  at `x = −1, 0, 1` center on `1.10, 1.99, 2.91` for a fit to `E[Y|x] = 2 + 0.9x`).

**Beta regression (#1608, #1609)**
- `gam sample` on a Beta-regression model aborted ("NUTS not implemented for
  beta-regression logit") instead of routing to the documented Gaussian Laplace
  fallback like every other NUTS-unsupported family. It now falls back correctly.
- `gam diagnose` computed Beta AIC / PSIS-LOO elpd at the placeholder precision
  `φ = 1` instead of the fitted `φ̂` (off ~1700 nats, incomparable across
  families). The fitted Beta `φ` is now threaded onto the reported family,
  mirroring the NB-`θ` path and restoring the `with_beta_phi` invariant.

**GAMLSS location-scale convergence (#1606, #1607)**
- Negative-binomial location-scale fits aborted with an `IntegrationError` on
  well-posed heteroscedastic counts: the log-`θ` dispersion block built its IRLS
  curvature from the strongly non-quadratic *observed* information, which goes
  negative for under-dispersed rows and divides the score by ≈0. It now uses the
  expected (Fisher) information, so the inner P-IRLS reaches KKT stationarity.
- The probit binomial location-scale outer REML/LAML curvature was likewise
  assembled from observed information, yielding an indefinite penalized Hessian
  that blew up the envelope-trace gradient (surfacing as an unavailable/zero
  analytic gradient). It now uses expected Fisher information.

**Flexible (learnable) link (#1596)**
- The flexible-link warp now genuinely engages and improves the fit: the frozen
  warp basis is de-aliased against the mean design (no canonical-gauge rank
  drop), the learned link is guaranteed strictly monotone/invertible over the
  fitted predictor range, and the warp is threaded through to `predict`. Deviance
  on the cloglog reproduction improves `1018 → 980` (reference `979.5`) with a
  certified monotone link.

**Multinomial REML invariance (#1587)**
- A penalized multinomial-logit GAM was not invariant to the arbitrary softmax
  reference class (predicted probabilities drifted ~1% under relabeling) because
  it applied the reference-anchored ALR penalty. The reference-symmetric centered
  CLR penalty `M⊗S_t` (`M = I − J/K`) is now wired through the custom-family outer
  REML loop; all other families are byte-identical.

**Duchon / Matérn smooths (#1604)**
- The half-integer-`ν` Matérn Taylor-coefficient path used the wrong polynomial
  degree (`2l` instead of `l`), collapsing every `ν ≥ 3/2` block (e.g. zeroing
  the `ν = 3/2` diagonal). Corrected, so `d = 1` hybrid Duchon smooths with power
  `≥ 2` build a PSD penalty again.

**Performance**
- Compensated multi-lane FMA GEMV kernels (faster *and* more accurate than the
  faer reference), truncated-Taylor `compose_unary` (~2.4×), SIMD-batched closure
  `design`/`design_jet` rows, output-symmetry Tower4 `t3`/`t4` contractions, a
  stable trig recurrence for the harmonic γ-jet, and a per-row alloc dropped from
  the `loss_scaled` data-fit hot loop. A measured survival regression from a
  build-once dense-3-tensor closure was reverted.

**SAE manifold (research surface)**
- Data-driven chart placement (`EncodeAtlas::build_data_driven`) places a bounded
  number of charts at the data's own latent coordinates (greedy farthest-point
  sampling), unlocking well-certified higher-dimensional manifold atoms; the
  PCA "flat SAE" baseline was replaced with a real trained TopK SAE; and the
  dense k-sweep "saturation" was diagnosed as a gate-cap bug and corrected.

**Release & build infrastructure**
- The intra-family dev-dependency publish cycle is broken so the crate family
  publishes cleanly in topological order (#1603); the `gam-terms` test crate
  builds clean again (#1601); broad new unit-test coverage across the
  predict / models / inference / terms crates.

## v0.3.129 — gam 0.3.129 / gamfit 0.1.231 (2026-06-28)

crates.io + PyPI release of the fix + SAE-fast-forward wave landed since gam
0.3.128 / gamfit 0.1.230. The most user-visible change is a prediction-contract
fix: `design_matrix(data) @ coef` now reproduces the reported `linear_predictor`
for every link (it was off by the bias-correction term for curved links). The
`gamfit` Python API surface is unchanged.

**Prediction contract (#1602)**
- The wiggle-free posterior-mean predict path reported a *bias-corrected* linear
  predictor `η̂_BC = X(β̂ + b̂)` (with `b̂ = H⁻¹S(β̂−μ)` the O(1/n) frequentist
  bias-correction) while the exported coefficients are the penalized MLE / mode
  `β̂`. That broke the documented "Raw design matrix" identity
  `design_matrix(data) @ coef == linear_predictor` (and the `posterior.samples @
  X.T` recipe) by exactly `X@b̂` — 1.5–4 % of the lp range for Poisson/Gamma log
  and binomial logit/probit, while staying exact only for the identity link. It
  now reports the uncorrected `η̂ = Xβ̂`, restoring the identity for all links and
  matching the plug-in / link-wiggle sibling paths.

**SAE manifold solver (#1033)**
- Frames-engaged SAE assembly (`build_framed_device_sae_data`, the decoder-rank <
  p large-output case) panicked at install: `set_device_sae_pcg_data`
  unconditionally asserted the per-row `a_phi`/`local_jac` slabs had length
  `rows.len()`, but the framed builder intentionally leaves them empty (the
  per-row cross block rides `frame.frame_blocks`). The length asserts are now
  gated on the non-framed path, so a real OLMo-shaped fit runs to completion. A
  regression test pins both the install (no panic) and the consumer contract (the
  CPU-resident reduced-Schur factor declines on empty slabs → generic matvec).

**SAE manifold structure search**
- `fold_atom_into`'s mass-preserving logsumexp combine produced NaN when fusing
  two zero-mass (`−∞`-logit) atoms (`−∞ − (−∞) = NaN`), poisoning the entire
  logits row and silently corrupting routing for every atom on it. Two zero-mass
  atoms have combined mass zero (logit `−∞`); that is now returned directly.

**SAE manifold fast forward (new public API)**
- A traditional-SAE-shaped GEMM forward pass for the manifold SAE:
  `EncodeAtlas::amortized_encode_batch_fast` / `amortized_reconstruct_batch_fast`
  (single atom) and the whole-dictionary LSH-routed
  `amortized_encode_with_index_fast` / `amortized_reconstruct_with_index_fast`.
  These run the routing + distilled affine predictor + curved-basis decode as
  batched matrix products (≈ a flat encoder's `W·x` throughput, the only extra
  cost the one batched basis eval), and were measured bit-faithful to the per-row
  predictor and accuracy-parity with the certified Newton solve. Degenerate rows
  (no evaluator / singular Gauss–Newton block / non-finite amplitude / no LSH
  proposal) are zeroed and flagged in a returned valid-mask — never a silent
  wrong encode/decode. The certified `*_encode_*` paths remain the accuracy mode.

**Testing & build**
- `cargo test -p gam-terms --lib` builds again (607 errors → 0, #1601): the
  #1521 carve left basis/smooth test fixtures referencing the pre-carve monolith;
  the basis fixtures are repointed at `gam-linalg` and restored, and the three
  smooth fixtures (which reach `gam-solve`/`gam-models`, below `gam-terms` in the
  dependency order) are set aside in `tests/src_modules/smooths/` for relocation
  to the top-level crate, tracked in the still-open #1601.
- Broad new unit-test coverage across gam-config, gam-data, gam-geometry,
  gam-gpu, gam-linalg, gam-math, gam-model-kernels, gam-models, gam-problem,
  gam-report, gam-runtime, gam-sae and gam-inference.

## v0.3.128 — gam 0.3.128 / gamfit 0.1.230 (2026-06-28)

crates.io + PyPI release of the open-issue fix wave landed since gam 0.3.126 /
gamfit 0.1.228. The most user-visible changes are the adaptive/flexible Binomial
link now fitting (or failing loudly instead of silently) and tensor smooths
becoming invariant to the typed order of their covariates; the `gamfit` Python
API is unchanged.

Versions skip gam 0.3.127 / gamfit 0.1.229: those numbers are already on
crates.io / PyPI from a prior orphaned release run whose version bump never
persisted to `main` (registry versions are immutable), so this release takes the
next free numbers — the same skip pattern recorded for 0.3.125 at 0.3.126.

**Adaptive / flexible Binomial links (#1596, #1598)**
- **#1598**: a `link(type=blended(...))` / `mixture(...)` learnable Binomial link
  is now fittable end-to-end. The Python/formula path threads the blended/mixture
  link components into the solver's `mixture_link` spec (it previously aborted
  before the solver with "BinomialMixture requires mixture_link specification"),
  and the joint link solve no longer refuses a finite-but-indefinite observed
  Hessian row: the array build was the lone over-strict consumer, and both
  downstream consumers already floor non-positive curvature, so the CLI fit that
  failed with "observed Hessian curvature is not positive finite" now converges.
- **#1596**: a non-convergent `link(type=flexible(logit))` wiggle fit is now
  surfaced **loudly** as an error instead of silently returning a model
  bit-identical to the fixed base link. Returning the large-smoothing baseline as
  if the flexible request were honored was a silent contract violation — callers
  could not distinguish a genuinely-flat learned link from a non-converged one.

**Gauge invariance (#1593)**
- A tensor-product smooth is now invariant to the typed order of its covariates:
  `te(x, z)` and `te(z, x)` span the identical tensor space under the identical
  per-margin penalty family, but the Khatri–Rao design permuted the columns and
  per-margin penalty blocks, routing the outer λ optimizer to a different terminal
  point in te's flat REML valley and drifting the shipped surface ~2–6 % of range
  on a cosmetic swap. Margins (plus feature columns and periods) are now
  canonicalized by source feature-column index at construction, so `te`/`ti`/`t2`
  build the identical problem regardless of typed order. Pinned by a new
  covariate-order regression test alongside the additive term-order, categorical
  reference-level, and by-factor labeling gauge guards.

**Survival (#1595)**
- `survival_at` and `cumulative_hazard_at` are now consistent past the fitted
  time grid: both flat-clamp beyond the support (they previously used contradictory
  right-edge extrapolation rules, breaking `S = exp(−H)` past the last grid point).

**GP / spatial smooths (#1074)**
- Isotropic Matérn / Duchon GP smooths now run a kernel-range multi-start: the
  profiled REML is re-fit across a log-κ grid and the strictly-best range adopted,
  so an unlucky single start no longer strands the fit on a poor local range.

**REML correctness (#1417 / #1006, #1038 / #1225 / #1418)**
- The REML log-det trace gradients now carry the full Daleckii–Krein
  deflation-derivative correction (with a divided-difference 0/0 guard),
  spectrally-deflated directions are excluded consistently across all four trace
  paths, and ungated atoms receive zero α-sensitivity in the data log-det trace.
- Streaming-exact REML now accumulates the cross-row IBP Woodbury log-det in both
  the criterion and the exact-Hessian matvec, matching the dense path to 1e-8; a
  non-PD capacitance is a recoverable ρ-probe refusal rather than a wrong number.

**Identifiability & competing risks (#1590)**
- The dead-column veto is narrowed to skip only entirely-zero placeholder designs,
  and channel-aware drop selection is now joint-rank-aware with faithful joint drop
  attribution for cause-specific competing-risks survival fits.

**Diagnostics & performance (#1575, #1557, #1151 / #1591 / #1592, #932)**
- The post-fit PSIS ρ-uncertainty diagnostic is now opt-in (default off), cutting
  ~33 surplus full-n solves per fit; the redundant parsimony second seed is waived
  for sharp well-penalized GLM optima.
- Extensive bit-identical jet/compose and SIMD row-batching speedups across the
  math, geometry, survival and SAE row kernels (straight-line Faà di Bruno /
  Leibniz towers, 4-row f64x4 lanes, pruned unused jet channels). The SAE
  arrow-Schur per-row GEMM is pinned to a sequential faer pool for
  parallelism-invariant losses.

**Validation & ergonomics (#1597, #11, #12, #13)**
- Weights-column validators report a **1-based** row index, matching the rest of
  the Python/data layer. Portable disk preflight in `build.sh`, a corrected
  `pip install torch` hint for `gamfit[torch]`, and an importable synthetic-SAE
  metrics bench round out the ergonomics fixes.

**Multinomial reference-class invariance (#1587, in progress)**
- Foundation toward making the penalized multinomial-logit fit invariant to the
  arbitrary reference class: the REML smoothing parameter is now **tied per term**
  across classes (the gauge the centered/CLR penalty requires), the
  reference-symmetric centered metric `λ·((I−J/K)⊗S)` is implemented and unit-proven
  in the vector-GLM engine, and a `CustomFamily::joint_penalty_specs` hook plus a
  `MultinomialFamily` centered-penalty builder are in place. The production formula
  path still uses the reference-anchored per-class metric pending the outer-REML
  joint-penalty wiring, so #1587 remains open; a red end-to-end repro documents the
  remaining drift.

**SAE structure & encode (#1026, #993)**
- Fission now applies an anti-symmetric decoder perturbation when it duplicates an
  atom, breaking the symmetric saddle that previously left the two children stuck
  in lockstep (so a bound product atom can actually split into its factors) while
  leaving the mass-split combined decoder exactly unchanged. The encode basin
  warm-up, NaN-alignment routing gate, and an opt-in outlier-robust per-row
  weighting policy for heavy-tailed activations also land.

**Build / CI**
- Completion of the public-API path restoration after the #1521 engine carve,
  GPU-kernel import repointing across the SAE FFI/examples/tests (#1577),
  line-count-gate decompositions (#780), repaired CI test-build APIs/paths, and a
  large set of new unit tests across the foundation crates (gam-problem,
  gam-linalg, gam-math, gam-spec, gam-geometry, gam-predict).

## v0.3.126 — gam 0.3.126 / gamfit 0.1.228 (2026-06-27)

crates.io + PyPI release of the open-issue fix wave landed since gam 0.3.124 /
gamfit 0.1.226, plus the completion of the #1521 engine carve so the whole
workspace — every crate library, the `gamfit` (gam-pyffi) wheel, and the
`gam` build.rs ban-scanner — compiles green again. The most user-visible change
is a correctness fix to the reported model-comparison statistics (AIC and
PSIS-LOO elpd); the `gamfit` Python API is unchanged.

**Reported log-likelihood / AIC / elpd (#1581, #1582, #1583)**
- The user-facing `log_likelihood` (and the conditional/corrected AIC and the
  PSIS-LOO `elpd` derived from it) is now the **fully normalized, scale-aware**
  log predictive density on the response's own measure, not the REML building
  block that deliberately drops every family normalizer and the Gaussian scale.
  New reporting kernels in `gam-solve` carry each family's full normalizer:
  Poisson `−lnΓ(y+1)`, Binomial `ln C(n, n·y)`, the Gamma saturated term, the
  Tweedie Jørgensen saddlepoint density, and Gaussian `−½[ln(2πφ̂) − ln wᵢ +
  wᵢ(y−μ)²/φ̂]` with the profiled `σ̂²` concretized into the scale (no silent
  unit-variance fallback). Symptoms fixed: a discrete model no longer reports a
  positive elpd (#1581); a Poisson fit and an NB(θ→∞) fit on identical data no
  longer differ by ~1750 nats (#1582); the Gaussian log density now obeys the
  change-of-variables law `elpd(c·y) − elpd(y) = −n·ln c` (#1583). An estimated
  dispersion now also adds its degree of freedom to the conditional AIC.
- The Binomial normalizer `ln C(n, n·y)` is now exact: `binomial_coefficient_f64`
  carries its multiplicative recurrence in integer (`u128`) arithmetic instead of
  dividing in `f64`, so the coefficient is bit-exact for every value at or below
  `2^53` (the prior all-`f64` recurrence drifted off the true integer well below
  that — e.g. `C(54,24)` came back one short, `C(55,25)` non-integer), keeping the
  reported Binomial log-likelihood / AIC / elpd exact.

**Survival & links**
- **#1569**: the post-update monotone-cone feasibility tolerance is floored at
  the same `1e-8` gate every downstream consumer enforces, so a cone-projected β
  feasible to the gate is no longer rejected by a stricter post-update check (the
  fragile spectrum-branch α-crush bypass was reverted after it could not be shown
  robust on the dense survival monotone cone).
- **#1571 / #1572 / #1573**: SAS / Beta-Logistic / mixture parameterized-link
  fits no longer abort with a "Lambda count mismatch": the post-convergence
  inner-cap guard now routes the augmented θ through the same `apply_link_theta`
  the eval closure uses, handing `compute_cost` exactly the smoothing-only ρ
  block (and installing the converged link state) instead of the raw augmented θ.

**Identifiability (#1580)**
- The large-scale identifiability-audit regression is rebuilt on orthogonal
  Legendre polynomials so its single seeded rank deficiency is resolved
  backend-independently (the penalty-augmented Gram path's `√ε` resolution made
  the prior RBF/trig fixture spuriously demote extra columns on some BLAS).

**Build system (#1521)**
- Completion of the engine carve: the published `gam` crate now depends on the
  full gam crate family (foundations plus `gam-model-api`/`gam-gpu`/
  `gam-identifiability`/`gam-terms`/`gam-solve`/`gam-custom-family`/
  `gam-model-kernels`/`gam-models`/`gam-sae`/`gam-test-support`), published to
  crates.io alongside it as a version-locked family; the `gamfit` wheel is
  unaffected (it builds from source). A sweep of latent carve breakages —
  cross-crate visibility, stranded duplicate definitions, a stale shared-include
  depth, mis-scoped GPU macros, and dead re-export shims — is repaired so
  `cargo check --workspace` is green end-to-end.

## v0.3.124 — gam 0.3.124 / gamfit 0.1.226 (2026-06-26)

crates.io + PyPI release of the open-issue fix wave landed since gam 0.3.123 /
gamfit 0.1.225. Two themes dominate: a survival/location-scale correctness pass
that makes saved-model prediction total and adds a genuine IPCW Brier score, and
the build-system work of #1521 — the monolithic `gam` crate is split into
foundation crates so an edit recompiles a sub-crate, not all 653 files. As a
consequence of that split the published `gam` crate now depends on the gam
foundation crates (`gam-runtime`, `gam-data`, `gam-math`, `gam-spec`,
`gam-linalg`, `gam-problem`), which are published to crates.io alongside it as a
version-locked family; the `gamfit` wheel is unaffected (it builds from source).
The pre-release hardening also restored the `cargo check --workspace
--all-targets` green invariant that two integration-test targets had regressed,
and removed leftover split-WIP scratch from the tree.

**Survival & location-scale**
- **#1564**: saved-model survival prediction is now total. The Royston–Parmar
  hazard guard accepts a zero log-cumulative-hazard time-derivative (the I-spline
  baseline is flat past its last interior knot, so `d(log Λ)/dt = 0` is a
  legitimate boundary value on the default prediction grid's top node) and
  resolves the saturated `Λ = +∞ × 0` corner to `0`, not `NaN`. A `finite_safe_json`
  serde adapter encodes `±∞`/`NaN` payload values as explicit string tokens so the
  engine→Python boundary no longer rejects non-finite `f64` as `null`.
- **#1563**: survival metrics now report a genuine integrated IPCW Brier score
  (Graf 1999) built on a Kaplan–Meier censoring estimator over a data-driven
  quantile grid, validated end-to-end against an independent Python oracle. The
  prior hazard-quadratic score is honestly renamed.
- **survival location-scale**: the log-σ (scale) design is kept raw instead of
  being residualized against the location design — a smooth that drives both the
  location and scale channels is separately identifiable, and residualizing
  erased the heteroscedastic signal and tripped a joint-gradient shape check on
  every smooth-scale fit. Cross-block identifiability is supplied by the
  per-channel audit assignment, matching the Gaussian location-scale path.
  Gaussian location-scale seed basins are also promoted/classified correctly.

**Manifold SAE (#1026, #1522)**
- **#1556**: manifold-SAE smoothness `λ_smooth` is genuinely per-atom (the outer
  ρ carries one coordinate per atom, not a shared scalar).
- Surplus/dead atoms are now parked gracefully instead of failing the pre-fit
  audit; per-atom ARD collapses to shared hyperparameters at large K; the outer ρ
  is routed through Fellner–Schall (REML); the over-complete reduced Schur is
  spectral-floored; and the large-K matrix-free regime is bounded by a wall-clock
  deadline so it cannot livelock. GPU device PCG for the SAE row-jet landed and is
  arch-pinned through NVRTC so the double-atomic kernels actually engage the
  device (#1017, #1033, #1551).

**GPU survival row-jet (#932)**
- An A100 survival rigid row-jet NVRTC kernel with a CPU-fallback dispatcher
  (≤1e-9 exactness), with device-fallback-reason logging and a device-only
  diagnostic entry.

**Inference, conformal & bases**
- **#1546**: the jackknife+ conformal interval uses `α = 1 − level` (delivered
  coverage), not `(1 − level)/2`.
- **#1548**: the default `s(x, bs="ps")` penalty is canonicalized so it is
  reflection-invariant; **#1549**: the ALR tangent coordinates are whitened by
  `G^{1/2}` so the smoothing penalty is Aitchison-isometric; **#1545**: the sphere
  Fréchet-mean Karcher descent is seeded from the full eigenbasis so the
  least-dominant axis is covered.
- **#1074**: `projected_gradient_norm` sign is corrected so a railed-but-descending
  ρ is not certified stationary.

**Python bridge & wheel**
- **#1565**: the `smooths={}` descriptor bridge is repaired (`slots=True`
  `super().__init__` across all sites; `double_penalty=False` is emitted).
- **#1559**: the `gam-pyffi` wheel build no longer fails on an `E0382` partial move
  (`log_lambda_smooth` is cloned instead of moved out of a still-borrowed ρ).
- **#1558**: the CUDA-unavailable diagnostic consumes `need_logdet` on every target.

**Build system (#1521) & release hygiene**
- The `gam` engine is split into foundation crates (`gam-math`, `gam-runtime`,
  `gam-data`, `gam-spec`, `gam-linalg`, `gam-problem`) plus the upper leaves
  (`gam-predict`, `gam-inference`, `gam-cli`), cutting per-change recompile from
  the full 653-file monolith to a sub-crate + facade. The families↔solver↔terms
  SCC stays in `gam`; its decomposition is tracked as separate contract-inversion
  work.
- Restored `cargo check --workspace --all-targets` to green: the `sae` and
  `perf_scale` integration-test targets had regressed against the post-split
  `gam::resource` module path and the per-atom `SaeManifoldRho` API. Removed the
  orphaned, unwired `gam-problem` `penalty_matrix.rs` staging file and the
  leftover split-WIP scratch notes from the tree.

## v0.3.123 — gam 0.3.123 / gamfit 0.1.225 (2026-06-24)

crates.io + PyPI release of the open-issue fix wave landed since gam 0.3.122 /
gamfit 0.1.224. The headline is the predict output layer: the reported
`std_error` and the `predict_array` return shape now match what `predict()`
documents and lays out, instead of silently handing back link-scale values on
non-identity links. Alongside that: low-cardinality cubic-regression smooths now
fit instead of hard-failing, the outer REML smoothing search is made invariant to
the order terms/margins are typed in, the default thin-plate basis is made
row-permutation invariant to the ulp, and the SAE held-out decode path is repaired
to match training. The build/CI hygiene also tightens (the full `tests/` suite now
runs in CI behind an orphan guard) and the workspace stays dead-code-clean as a
primary build (`cargo check --workspace --all-targets` green, so the published
`gam-pyffi` wheel's `use gam::…` surface is verified, not just the `gam` lib).

**Prediction output layer (response scale)**
- **#1536**: `predict(interval=...)` / `gam predict` now report `std_error` on the
  *response* scale — the delta-method `SE(μ̂) = |dμ/dη|·SE(η)` the credible band
  beside it is built from — instead of the link-scale `σ_η`. On a non-identity
  link the two were off by the inverse-link Jacobian, so the SE column was
  internally inconsistent with its own `mean`/`mean_lower`/`mean_upper`. The
  posterior-mean path gained a `PredictPosteriorMeanResult::mean_standard_error`
  field, populated from the SE the band already uses and surfaced by the FFI/CLI.
- **#1537**: `predict_array(X)` with no `interval` now returns the documented 1-D
  response-scale vector, matching `predict()`, instead of the 2-D
  `[linear_predictor, mean]` column matrix — a naive `[:, 0]` / `.ravel()` caller
  was silently getting the link-scale linear predictor on non-identity links. The
  interval case still returns the full column matrix.
- **#1515**: interval predict on a degenerate fit no longer returns non-finite
  bounds. When the smoothing-corrected covariance `H⁻¹ + J Var(ρ̂) Jᵀ` carries
  non-finite entries (e.g. an all-zero-count Poisson, whose flat likelihood leaves
  the outer REML problem near-singular and blows up `Var(ρ̂)`) the predictor now
  degrades to the finite conditional covariance `H⁻¹` — the `Preferred` mode
  already falls back when the correction is *missing*, and an unusable (non-finite)
  correction is the same case — so a model the API reports as fitted always yields
  finite `std_error` and `mean_lower`/`mean_upper`.

**Smooths & bases**
- **#1541**: a univariate `s(x, bs="cr"/"cs")` cubic-regression smooth no longer
  hard-fails the whole fit on a low-cardinality covariate (a binary indicator, a
  3-level ordinal, a small count). The basis is capped to the data support —
  `k = min(k_requested, n_distinct)` value-knots, mgcv-style — and below the cr
  minimum of three distinct values it degrades to the linear B-spline marginal the
  default `s(x, k=..)` basis already builds. The cap is surfaced in the inference
  notes. This is the univariate sibling of the tensor-margin cap.
- **#1542**: a factor smooth `s(x, g, bs="sz")` likewise caps its per-level cr
  marginal to data support rather than aborting. A pre-existing latent bug was
  uncovered and fixed in the same change: a frozen `sz` factor smooth failed its
  own predict-time freeze check ("factor-smooth marginal knots missing") because
  the validation whitelist had never been updated for the cr marginal — so `sz`
  *predict* was broken regardless of cardinality. It now fits *and* predicts.
- **#1543**: a basis the fit silently adjusted is no longer silent to gamfit
  callers. The mgcv-style cap/degradation advisories the Rust core records (and
  the CLI already prints) are now carried through the FFI to Python: `gamfit.fit`
  / `fit_array` emit one `GamInferenceWarning` per note at fit time, and
  `Model.notes` exposes them for after-the-fact / post-load inspection. Previously
  the Python path dropped `inference_notes` at the FFI boundary, so a capped basis
  warned in the CLI but was invisible to gamfit. (The payload field is
  `#[serde(default)]`, so older saved models load cleanly as "no notes".)
- **tensor margins**: explicit `te(...)` margin `k` is capped to data support
  (mgcv-style), with the `cr` `basis_size` helpers repaired.
- **#1378**: the default `s(x, bs="tp")` thin-plate smooth is made exactly
  row-permutation invariant. The knot-selection centroid was summed in row order,
  so floating-point round-off shifted it by an ulp under a pure row permutation —
  enough to flip the seed (and hence the whole knot set and `λ̂`) on data symmetric
  about the mean. The column sum is now taken in canonical value-sorted order.

**REML smoothing-parameter selection**
- **#1538 / #1539**: the outer REML smoothing-parameter search is made invariant
  to the order terms and margins are written in — additive `s(x)+s(z)` vs
  `s(z)+s(x)` and tensor `te(x,z)` vs `te(z,x)` now select the same `λ̂` and fit
  the identical surface (worst row-drift 1.0e-1 → 1.8e-6, 6e-2 → 7.1e-5). Each
  rho-coordinate is labelled by a placement-independent canonical key (the
  penalty's orthogonal-invariant spectrum plus a data-dependent block signature),
  so seeding, multistart and tie-breaking all run on one canonical layout and map
  back to the native order for the caller. Single smooths run the native path
  byte-for-byte as before.

**SAE (sparse autoencoder)**
- **#1540**: the held-out SAE reconstruction now attaches the trained dictionary's
  hybrid-collapsed straight images, so verdict-linear `d=1` slots decode by the
  same linear image training used. The parameter was accepted but never wired, so
  every OOS reconstruction silently fell back to the all-curved decoder — a
  train/test decode mismatch on hybrid-collapsed dictionaries.
- **#1026**: SAE anti-collapse interior-point barriers (finite-difference
  verified), runtime barrier-strength and IBP-alpha overrides so one wheel sweeps
  all configs, and GPU residency Phase 0-1 telemetry / fail-closed wiring.

**Survival**
- **#979**: the marginal-slope seed-screening cascade is bounded by the outer
  wall-clock deadline (single-sourced with the slow-geometric-rate stall guard),
  so a hard survival fit cannot blow the outer time budget in the inner search.

**Model summaries**
- **#1544**: `MultinomialModel.summary()` (and `str()`/`print()`) no longer raises
  `ValueError` on a smooth multinomial fit. The summary assumed one λ per smooth
  term per class, but the default Marra–Wood double penalty emits two penalty
  components — a wiggliness penalty plus a null-space shrinkage penalty — so the
  per-block λ count never matched the term-label count. The summary now records
  per-penalty-component λ labels and pairs every component (including each term's
  null-space λ) instead of silently dropping it.

**Custom families & outer-score subsampling**
- A custom-family / `GaussianLocationScale` fit with a Horvitz–Thompson
  `outer_score_subsample` no longer hard-fails with `IntegrationFailed`. The inner
  coefficient solve was mixing two row measures — a full-data entry/reload base
  objective against an HT-subsampled trial — for families that do not advertise an
  HT-consistent inner gradient, so the trust-region `actual_reduction` was pinned
  at the constant HT-vs-full log-likelihood gap, the radius collapsed, and every
  seed was rejected at outer startup. The subsample is now stripped from the inner
  options unless the family runs a fully HT-consistent inner solve, keeping β̂(ρ)
  the unbiased full-data optimum (the subsample remains an outer ψ/ρ-derivative
  variance-reduction device). Covered by the `ws4a` subsampled-vs-full parity test.

**Build & CI hygiene**
- **#1534**: the manylinux / musllinux wheel containers trust the mounted
  workspace tree (`git config --global --add safe.directory '*'`) so `build.rs`'s
  author gate runs instead of panicking on "dubious ownership" ~12 min into the
  release build.
- **#1512**: the full `tests/` suite now runs in CI via a directory-level pytest
  step, behind a hard-gated orphan-guard meta-test, so new `test_*.py` repros are
  collected automatically instead of silently running in no job.
- **#932 / #1017**: continued survival flex single-source derivative cutover onto
  the jet tower and removal of the dead dual-context CUDA path, keeping the `gam`
  crate dead-code-clean as a primary build.

## v0.3.122 — gam 0.3.122 / gamfit 0.1.224 (2026-06-24)

crates.io + PyPI release of the open-issue fix wave landed since gam 0.3.121 /
gamfit 0.1.223. A robustness sweep across degenerate-fit prediction and
constrained-coefficient posterior sampling, family deviance/dispersion
corrections, a pure-REML escape for double-penalty null-space shrinkage, further
thin-plate/Matérn root-cause cleanup, the survival flex single-source derivative
cutover (with the fourth-order moving-boundary residual further closed), and SAE
collapse *prevention* — a data-derived bar plus a linear-dominance floor in place
of magic constants. The build.rs hygiene gate continues to hold: spent #1454
localizer probe fields are removed and the unwired survival moment-engine oracle
is relocated into its test module, so the `gam` crate stays dead-code-clean as a
primary (`cargo build` / `cargo publish`) build.

**Prediction & posterior sampling**
- **#1515**: degenerate / near-singular fits no longer emit non-finite
  predictions. Interval predict gets a finite delta-method SE fallback, the
  log-link posterior mean is floored when its SE overflows, and an all-zero
  Poisson fit predicts a finite plug-in mean when the posterior-mean integral
  overflows — with finite interval bounds throughout.
- **#1507 / #1509**: box- and shape-constrained coefficients now draw from a
  truncated-Gaussian posterior on the latent scale, so `predict_draws()` respect
  the monotone-shape and `bounded()` box constraints instead of escaping them;
  posterior-predictive draws also re-apply the model offset.
- **#1514**: a `bounded()` Gaussian coefficient covariance is scaled by σ̂², so
  its standard errors are neither too wide nor too narrow.
- **#1513**: numeric `by=`-variable multipliers are exempt from the predict-time
  axis clip (they are multipliers, not a smooth axis).

**Families & model comparison**
- **#1529**: Gamma deviance-explained is no longer contaminated by an
  estimated-shape mismatch — the null deviance uses the fitted dispersion rather
  than resetting to the family default.
- **#584**: ALO dispersion divides by the positive-weight row count (the true
  residual dof), not the raw row count, so zero-weight rows do not deflate it.
- **model comparison**: the smoothing-correction covariance `V_corr` is
  symmetrized before use (#1527).

**Smooths, REML & double penalty**
- **#1266**: the default double penalty (Marra–Wood null-space shrinkage) no
  longer inflates smooth EDF — a pure-REML null-space shrink-out escape lets an
  irrelevant covariate be shrunk out (and `s(x)` on linear data recover its true
  ~2 EDF) instead of pinning every term near the basis dimension.
- **#1074**: further thin-plate / Matérn root-cause cleanup — the masking hacks
  (the Matérn-specific length-scale ceiling, a redundant λ ceiling, the
  thin-plate cap and center-cap, the latent active-mass floor) are deleted in
  favor of the real basis-sizing / correlation-range fixes; `te()`/`ti()` cr
  margins honor the requested `k` (guarded to `k ≥ 3`) and `bs="sz"` factor
  smooths route through the cr metadata freeze/replay.
- **#1531**: the constant-curvature double penalty is documented and tested to
  use an identity ridge (full-rank null space). **#1464**: the curv
  hyperbolic-sign contract is now an asserting CI gate.

**Survival (#932 / #1454)**
- **#932**: the survival flex marginal-slope derivative tower is single-sourced
  through one generic `FlexJet` jet algebra — the link-wiggle joint-Hessian
  cutover is landed in production and the generic-order moment / eta-chi
  machinery is exact to all carried orders, with the calibration residual derived
  as a distinguished-derivative `j/(j+m)` projector.
- **#1454**: the fourth-order `[g, β_w]` bidirectional cross residual is
  corrected in sign and magnitude — the moving-boundary `D²(B)` term and the
  missing `f_a` self-flux are added to the bidirectional `§D` path. The final
  observed-point link-warp term remains and **#1454 stays open** for it.

**Manifold-SAE & latent (#1026 / #1388 / #1522)**
- **#1522**: the SAE collapse **floors** (a magic 0.28 reconstruction-EV bar, a
  1e-3 atom-mass floor, a latent active-mass floor) are replaced by a
  data-derived PCA-EV ceiling plus a genuine NaN guard — collapse *prevention*
  in the assignment/decoder step rather than detect-and-reseed band-aids.
- **#1026**: a hybrid-split collapse rescue rebuilds a fresh linear image for
  rank-1 co-collapsed circle atoms, and a result-level linear-dominance floor in
  `into_fitted` restores the certified PCA anchor (`F ≤ F_linear`) when curvature
  collapses — backed by a collapse-safe SAE acceptance battery.
- **#1388**: the SAE joint fit runs on a wide-stack worker thread in the wheel,
  avoiding a stack overflow on large joints.

**GPU & build (#1017)**
- **#1017**: CUDA initialization is hardened — the primary context is bound and
  the runtime initialized before the first cuBLAS/cuSOLVER handle creation
  (probe-first `NOT_INITIALIZED`), and the probed compute libraries stay loaded
  so `dlclose` cannot poison cuBLAS; a GPU regression guard and a Modal A100/T4
  runner (with a dead-hand heartbeat kill-switch) back it.
- **CI**: the full pytest suite runs in CI (#1512 / #1532) and the fast Python
  Contracts workflow caches `target/` and enables sccache (#1518).

## v0.3.121 — gam 0.3.121 / gamfit 0.1.223 (2026-06-23)

crates.io + PyPI release of the open-issue fix-and-feature wave landed since gam
0.3.120 / gamfit 0.1.222. New data-ingestion surface (Dask, SPSS, numeric-string
categoricals), a large correctness sweep across double-penalty smooths, REML
convergence, survival/location-scale Hessians, model comparison and the
manifold-SAE / latent stack, plus opt-in routing-predictor machinery and the
device-resident GPU Gram path. The build.rs hygiene gate is hardened further
(anti-laundering bans) and the tree continues to build the `gamfit` wheel clean.

**Data input (Python wrapper)**
- **#1460**: Dask DataFrames are accepted as input/output — materialized with a
  single `compute()` into pandas for fitting — and a new `read_spss()` loads
  `.sav`/`.zsav` files via `pyreadstat`, decoding value-labelled variables to
  pandas `Categorical`.
- **#1467 / #1468 / #1469 / #1473**: numeric-string columns in dict / records /
  numpy inputs are treated as categorical (pandas-object parity); a mixed
  string+numeric column is categorical, and dict-input multinomial hard-rejects
  numeric-string class labels.

**Smooths, factors & identifiability**
- **#1476 / #1477**: the double-penalty null-space ridge is rebuilt in the
  identifiability-constrained chart (after the global transform), pairing each
  ridge with its co-located Primary block — fixing concurvity collapse of
  `s(x1)+s(x2)`, by-factor per-level correctness, and Tweedie default-P-spline
  mean bias, without over-shrinking a supported smooth.
- **#1427**: `s(x, by=factor)` emits an independent per-level λ. **#1457**: a
  bare categorical main effect is de-duplicated under `s(x, by=g) + g`.
- **#1470**: `ti(x, z)` interactions stay grid-independent — no off-grid
  residualization against the realized `s(x)`/`s(z)` spans in functional-ANOVA
  models.
- **#1378 / #1456**: default univariate thin-plate basis sized to mgcv `k=10`
  with rotation- and permutation-invariant knot selection. **#1074**: default
  thin-plate / Matérn basis sized to mgcv `k=10·3^(d-1)` and the Matérn
  correlation range matched to mgcv's default (diameter), fixing EDF inflation.
- **#1379**: per-block penalty trace clamped to `[0, rank]` (NaN-safe) so a
  univariate `matern(x)` fits. **bs="sz"**: emits `FactorSmooth` metadata so
  basis freezing matches the spec.

**Inference & numerics**
- **#1426**: a stuck gamma/log REML flat valley is no longer shipped as a
  converged overfit — score-relative stationarity certification, rejection of
  non-converged inner PIRLS iterates and untrustworthy release-rerank seeds, and
  a rank-guarded `H` pseudo-logdet with a determinant-pair-sign guard.
- **#1464 / #1404**: constant-curvature κ sign is identifiable in `curv()` — a
  κ-fair scan recovers the hyperbolic/spherical sign and the joint solve is
  pinned to the sign-correct half-axis; the curvature-blind double-penalty ridge
  is dropped.
- **#1395**: custom-family pseudo-Laplace / exact-Newton objectives gain a
  structural guard against `0.5·log|H|` collapse and no longer fold the
  Jeffreys/Firth prior into the pinned-mode objective. **#1418**: the IFT
  back-substitution inverts the exact stationarity Jacobian.
- **#1376**: anisotropic Matérn ψ- and penalty-second-derivatives centered to the
  raw-η gauge. **#1392**: P-spline double-penalty underfit fixed for `p>n`.
- **#1410 / #1419**: compact softmax curvature uses a genuine Gershgorin Loewner
  majorizer and reads active-only entries. **multinomial**: genuine per-class
  EDF (no per-penalty-block over-count), a Fisher-information sparse-class
  λ-floor (#1082), and the hetero `x1` basis sized to its true df (#1373).

**Families & model comparison**
- **#1465**: `compare_models` computes Δ / Bayes-factor on the ranking scale.
- **#1448**: negative-binomial runs the full outer θ↔λ alternation (re-selecting
  ρ after each θ refresh); **#1463**: NB-NUTS `sample()` refreshes the fitted
  `theta_hat` rather than the seed.
- **#1504**: a Gaussian location-scale (gaulss) fit with a by-group smooth in
  both the mean and log-σ blocks no longer crashes on a joint-Hessian shape
  mismatch — the joint exact-Newton path uses the identifiability-constrained
  designs (with an R-free regression guard).

**Survival**
- **#1454**: the survival flex intercept-Hessian moving-boundary / self-flux
  terms are completed and carried exactly to fourth order, single-sourced from
  the D-path. **#1396**: entry/exit transposition in time-block η slicing fixed
  and a near-cancellation event-Jacobian floored to the monotonicity guard.
- **#1388**: under-determined (`p_joint > n`) survival marginal-slope joints are
  surfaced honestly. **#979**: the marginal-slope outer search is bounded by a
  wall-clock deadline with collapsed-trust-region stuck-exit guards.

**Manifold-SAE, latent & routing (#932 / #1017 / #1026 / #1033)**
- **#932**: the survival-LS / BMS-rigid / SAE-β-border row kernels are
  single-sourced through one `row_kernel` v/g/H tower with an exact wiggle
  joint-Hessian oracle. **#1500**: dead dictionary atoms are re-seeded.
- **#1017**: a device-resident GPU path uploads `X` once and chains a
  Gram-resident POTRF, downloading only β (with CPU parity gates). **#1026**: an
  ungated linear/background tier reconstructs full-rank alongside the gated
  sparse atoms.
- **#1033**: an opt-in chart-geometry / amortized routing predictor (off by
  default) with an n-free frozen-W Fisher-step solver. **latent**: a new
  `LatentIdMode::IsometryToReference` gauge-fix mode.

**CLI & summaries**
- The CLI honors formula-declared categorical roles (numeric-coded factors).
- **#1368 / #1370**: `summary()`'s RE penalty-cursor skips empty-range penalized
  RE blocks on both the in-process and Python persisted paths.

**Releasability / hygiene**
- The build.rs gate gains anti-laundering bans: silent NaN/`0.0`/`Ok(())`
  corruption where a contract guard belongs, owed-work disguised as prose, and
  hardcoded commit-SHA literals. Contract guards previously laundered into
  silent corruption are restored to panics-with-`// SAFETY:` or proper `Result`
  errors across penalties, solver, basis, evidence, families and GPU paths.
- PR-level anti-evasion / owed-work-ledger workflows are removed (enforcement
  lives in the build), clippy is dropped from CI, and **#1458** gives the
  build.rs author gate full history so it resolves the real last editor.

See the git history (`git log v0.3.120..v0.3.121`) for the complete set.

## v0.3.120 — gam 0.3.120 / gamfit 0.1.222 (2026-06-21)

crates.io + PyPI release of the open-issue bug-fix wave landed since gamfit
0.1.221 (gam 0.3.119), plus the #1452/#1288 releasability cleanup that re-arms the
`build.rs` hygiene gate to a hard failure and brings the tree into compliance so a
release can actually build the `gamfit` wheel.

**Releasability / hygiene (#1452 / #1288 / #780 / #871)**
- The `build.rs` ban-scanner is back to a hard `exit(1)`: no `#[ignore]` tests,
  no `debug_assert!`, no `unreachable!`, no manifest `dead_code = "allow"`, no
  underscore-prefixed unused parameters, no `#[cfg(test)]` items outside a test
  module, and no tracked file over 10k lines. The whole tree now complies and
  builds clean under `[lints.rust] warnings = "deny"`.
- Finished the #1288 dead-code cleanup: removed the never-consumed third-order
  `d3qdot` location-scale qdot jet and the unwired batched
  `jeffreys_*_flex_no_wiggle` / basis-contraction survival fast paths, scoped the
  #932 jet-scalar oracle structs to their test module, and dropped leftover
  #932 directional FD-localizer debug scaffolding.
- Promoted cheap invariant `debug_assert!`/`debug_assert_eq!` to always-on
  `assert!`/`assert_eq!` (penalty-root rank consistency, Duchon hybrid-integral
  precondition); dropped release-noop debug assertions where the documented
  contract wants honest IEEE `NaN`/`inf` propagation; replaced a banned
  `unreachable!` with a `// SAFETY:`-justified `panic!`.

**Inference & numerics**
- **#1436**: a typed `OuterGradientError` (IllConditioned / NonIdentifiable /
  InternalInvariant) narrows the SAE FD-fallback so only genuine
  conditioning/identifiability failures admit the finite-difference descent
  direction at a finite-cost ρ, while internal-invariant defects propagate as
  hard errors. NonIdentifiable is now constructed at the gauge-degenerate,
  non-deflatable outer-gradient site.
- **#1424 / #1422 / #1423**: cancellation-free hybrid Duchon-Matern kernel
  evaluation and a PSD mixed-periodicity Duchon penalty via an additive tensor
  reproducing kernel, with the correct cylinder nullspace.
- **#1271 / #1266 / #1380 / #1089**: the REML log-λ cap is lifted off
  well-determined Gaussian-identity smooths so REML can reach the null-space
  optimum, without changing the global default.
- **#1391 / #1397 / #1017**: post-T rank invariant is anchored to the audit's
  kept-rank certificate and made robust to the drop-deciding convention;
  relaxed over-strict arrow-Schur parity asserts to a tight relative tolerance.
- **#1376 / #1398 / #1404**: anisotropic Matern ψ-derivative centered to the
  raw-η gauge; isotropic sphere-harmonic penalty and closed-form Sobolev jet
  with a constant-curvature effective-length contract.
- **#1426**: a stuck gamma/log REML stall is no longer shipped as a flat valley.
- **#1385**: competing-risks CIF assembled on a refined internal grid.

**Survival (#932 / #979 / #1394 / #1396)**
- Moving-boundary flux / implicit-function / substitution jet-tower combinators
  for the θ-dependent flex-calibration integrand, carried exactly to fourth
  order, with the survival location-scale time-channel NLL sign living in a
  single source of truth and rigid-kernel non-finite margin propagation.

**Smooths, factors & summaries**
- **#1403**: `s(x, by=factor)` routes to `BySmooth::Factor`, `s(x, by=numeric)`
  to `BySmooth::Numeric`, and `bs="sz"` factor smooths to `FactorSmooth { Sz }`.
- **#1378**: default univariate thin-plate basis sized to mgcv `k=10`, with
  row-permutation-invariant knot selection.
- **#1364**: P-spline scale equivariance.
- **#1384**: `compare_models` refuses to rank fits of different response families.
- **#1370 / #1368 / #1369**: `summary()` synthesizes valid factor levels for the
  smooth-term replay so `fs`/`sz`/`by` smooths keep their EDF and per-level
  labels.

**Manifold-SAE**
- **#1405 / #1406 / #1410 / #1411 / #1412**: matrix-free planner predicts the
  true cross footprint; compact support is partial-selected and per-worker
  scratch sized by compact dims; encode bench gates honest support recovery.
- **#1026**: collinearity-gated decoder repulsion conditions the SAE
  co-collapse direction with a keep-best multi-start.

**Error messages**
- **#1445**: NaN/None/empty table-cell errors now name the offending column and
  row instead of an unactionable bare message.

See the git history (`git log v0.3.118..v0.3.120`) for the complete set.

## v0.3.118 — gam 0.3.118 / gamfit 0.1.219 (2026-06-17)

crates.io + PyPI release. The `gam` crate is bumped 0.3.117 → 0.3.118: this is a
crates.io catch-up that publishes to the `gam` crate all the engine work that has
shipped to the `gamfit` wheel since gam 0.3.117 (gamfit 0.1.203), plus the open-
issue bug-fix wave landed since gamfit 0.1.218. Highlights:

**Basis & boundary conditions**
- **#1238**: B-spline `bc=anchored`/`bc=clamped` endpoints are now enforced as a
  *structural* nullspace reparameterization — anchored endpoints are pinned to
  zero (and drop their constrained column), clamped derivatives are zeroed, and a
  non-zero anchor is rejected. The free intercept is suppressed only for a
  *one-sided* anchor (which consumes the absolute level at that endpoint); a
  two-sided anchor keeps the intercept.
- **#1239**: periodic B-splines evaluate the derivative recurrence on the full
  wrapped knot support, no longer extrapolate past a clamped boundary, and drop
  the ridge fallback that masked the wrap.
- **#1257**: `periodic(x, …)` is accepted as a term-function alias and routes to
  `cyclic`. **#1132**: periodic/torus `n_harmonics` floored at the decoder-
  implied harmonic count.

**Manifold-SAE (large body of work: #977 / #1026 / #1132 / #1154 / #1189–#1232)**
- Chart canonicalization ordering/turning stabilized and the hybrid curved-vs-
  linear split computed *after* canonicalization (#1227); OOS fixed-decoder solve
  returns the converged latents (#1229); shape uncertainty recomputed after the
  structure search settles (#1230); hybrid-collapsed linear images threaded into
  OOS reconstruction (#1228).
- Outer objective fixes: BFGS/ARC line-search probe sees pure REML, not `f+c`
  (#1224); streaming SAE branch optimizes the full REML criterion (#1225);
  consistent cost/gradient pair in the cotrain outer objective (#1206/#1207);
  PSD softmax Fisher metric for the curvature anchor (#1190); corrected PG
  gate-block normalizer in the live K-vs-K+1 birth gate (#1218).
- n-free per-ψ penalty rebuild + ψ-Gram certification on standardized geometry
  (#1033 / #1216); EV-knee auto-K + manifold-vs-linear wager verdict (#977 /
  #1026); honest EV/centering and labeling throughout (#1198 / #1201 / #1202 /
  #1203 / #1209 / #1213 / #1226).
- **#1232**: SAE top-k projection metadata preserved in the Python payload;
  per-atom `held_out_delta_ev` and the (Θ, ΔEV) frontier surfaced through the FFI.

**Survival**
- **#931**: robust inner-solve polish (regularized-Newton + steepest-descent +
  Armijo value line-search + exact Cholesky) reaches stationarity at all ρ and
  large λ; survival LAML objective↔gradient desync closed via the active-set-
  projected IFT envelope.
- **#740**: full-θ KKT-residual correction (cross-ρ-ψ + ψ-ψ Hessian); binomial
  loc-scale drift FD on the identifiable Jeffreys span. **#1242**: derivative-
  channel location-scale row derivatives aligned with the exact-Newton kernel.
- **#1248**: `survival_likelihood` canonicalized consistently across CLI and JSON
  config paths. **#1258**: consistent `expected 0 or 1` event-target message.

**Gaussian / GLM / conformal**
- **#1262**: an effectively-constant Gaussian response now fits to the exact
  constant instead of erroring out of the REML path.
- **#1261**: Gaussian average-derivative one-step debiased against the unpenalized
  information (sign + smoothing-pull corrections). **#1127**: scale-equivariant
  REML smoothing-parameter selection.
- **#942 / #1098 / #1192 / #1263**: GLM full-conformal route contract stabilized
  (KKT-scaled cold-fit/corrector convergence, round-off-floor cold fit, alternate
  ARC seed retained after screening).

**Inference**
- **#939**: Skovgaard r* modified directed-likelihood root for scalar contrasts
  (matrix-level assembler + ρ̂-variation Bartlett factor + >10% material flag).
- **#1219**: per-term EDF for `te()`/`ti()` is the influence-matrix trace over the
  term's coefficient block (was double-counting shared tensor coefficients).

**Terms, families & observation bands**
- **#1064**: `--family gamma` accepted as an alias for `gamma-log`. **#1160**:
  `Smooth(by=col)` plumbed through the `smooths={}` descriptor path. **#1158 /
  #1159**: marginality-aware `:` interaction expansion. **#1214 / #1215**:
  covariate-rescaling invariance for `cr` and `tp` smooths. **#1246**: sphere
  `wahba_sobolev`/`wahba_pseudo` aliases + `degree=` route to `harmonic`.
- **#1193 / #1194** (with #817): equal-tailed Poisson, Tweedie, NB and Beta
  observation bands.

**Numerics & performance**
- n-free κ-loop fast path with bit-exact β̂ on the slow path (#1216); on-device
  CUDA Step-6 joint-β contraction for survival-flex (#1133); per-block
  λ-coercivity threshold in the penalty pseudo-logdet (#1237); flat-residual
  stall now exits the inner joint-Newton instead of grinding to the cycle cap
  (#1040); exact tall RRQR on exact-collinearity (#933); ALO stabilization
  degrades gracefully instead of aborting the outer eval (#1191).

(See the 0.1.217 / 0.1.218 entries below for the GPU joint-Hessian build fix and
the BLAS-3 `coord_corrections` perf sweep that this crate release also carries.)

## gamfit 0.1.218 (2026-06-15)

- **Build fix**: wire the whole-`Xᵀdiag(w)X` GPU joint-Hessian path
  (`rigid_joint_hessian_on_gpu`) into the rigid `hessian_dense_override`, guarded
  by a GPU-presence probe (CPU boxes skip the weight-vector alloc and take the
  chunked-BLAS3 path unchanged). Clears the dead-code ban that failed 0.1.217.
- Saturate the rayon pool in `chunked_row_reduction` (4×workers chunks, was a
  fixed 32 that idled half a 64-core box) — completes the CPU-utilization fix.
- `DenseDesignMatrix::cache_identity` for memoizing the `X·F` projection across
  the k per-coordinate correction operators within one outer eval.

## gamfit 0.1.217 (2026-06-15)

- **Build fix**: re-export `SaeBasisEvaluator` in the pyffi prelude. The #1117 SAE
  fix retyped evaluators to `dyn SaeBasisSecondJet`; calling its supertrait
  `.evaluate()` needs `SaeBasisEvaluator` in scope, which broke the 0.1.216 wheel.
- Perf sweep continues: n-independent outer loop (eliminate redundant ext-coord
  n-row drift re-streams), SIMD-batched rigid per-row jet, GPU-routed rigid
  `XᵀWX` Gram when CUDA present (CPU fallback otherwise), cross-disease duchon
  basis+identifiability cache (build once for the shared cohort, not 17×), and
  the BLAS-3 batched all-axes second-directional override (~p× on the dominant
  coord_corrections term).

## gamfit 0.1.216 (2026-06-15)

Open-issue bug fixes + inner-solve perf:
- **#1128**: `gamfit.fit(Surv(...))` with no `survival_likelihood` now defaults to
  `transformation` (matching the CLI) instead of the broken `location-scale` that
  aborted the identifiability audit on right-censored data. Fixed at the single
  `FitConfig::default()` source.
- **#1127**: Gaussian `s(x)` REML is now scale-equivariant to `y→a·y` — the
  singularity floor `smooth_floor_dp` is a fraction of the weighted null deviance
  (was absolute 1e-12), so `λ̂`/EDF/smooth-shape are invariant down to a=1e-8.
- **#1117**: the SAE production term builder now installs the analytic second jet,
  so a rank-deficient K=1 circle decoder reparametrizes to its data-supported rank
  and completes stage1-step0 in budget instead of stalling.
- **#1126** (already on main): measure-jet κ non-convergence degrades to the
  certified baseline geometry instead of fatally aborting at tol=1e-10.
- **PERF**: the inner DENSE_SPECTRAL joint-Newton path no longer re-applies the
  matrix-free operator ~25× per cycle (trust-region model + Cauchy leg) — those
  route through the already-materialized dense Hessian (O(n·p)→O(p²)).
- Removed a second per-outer-eval `log::debug!` spam site (gated to once/process).

## gamfit 0.1.215 (2026-06-15)

- Remove per-call `[STAGE] BMS rigid ... BLAS-3 ... path TAKEN/NOT-taken` log
  lines from the rigid row-kernel dispatch (added in 0.1.213 for one-shot gate
  diagnostics). They fired on every `directional_derivative`/`hessian_dense`
  call — thousands of lines per fit — flooding the run log. The gate logic is
  unchanged; only the logging is removed.
- Cross-fit warm-start descriptor now encodes the realized per-block reduced
  β-width, so a p=37 fit no longer matches a p=85 artifact (no misleading
  length-mismatch skip) while same-width LOSO folds still transfer β.

## gamfit 0.1.214 (2026-06-15)

Biobank BMS speed sweep — attacks every recurring cost in the outer REML/LAML
loop, all exact (bit-faithful, no approximation, no skip flags):
- **coord_corrections** (the ~1.5–4min/eval Jeffreys H_phi drift): β-fixed base
  hoisted out of the per-direction loop + both p-axis row-stream sweeps
  parallelized across cores.
- **gradient_reload** (~5s/inner-cycle): the accepted trust-region line-search
  workspace is now reused, collapsing each accepted cycle from two row passes to one.
- **Murphy–Topel** SE correction and the **latent-z Rao-gate** score+meat: per-row
  scalar scatters replaced with single BLAS-3 GEMMs.
- **identifiability audit**: joint RRQR now runs from a single shared Gram (was a
  second full n-row stream); trivial full-rank case skips the redundant pass.
- **FFI encode**: column-major borrow (no StringRecord clone), parse-once, and a
  content-fingerprint cache so the shared base cohort is encoded once across diseases.
- **outer BFGS eval count**: the converged outer Hessian is transferred across LOSO
  folds to seed quasi-Newton, cutting line-search probes.
- **large-p outer LAML logdet**: one-pass dense assembly instead of p matvecs.
- BLAS-3 rigid Hessian fires for operator-backed designs; warm-start cross-fit
  length-mismatch declines to ρ-only instead of cold-starting.

## gamfit 0.1.213 (2026-06-15)

Continues the biobank BMS perf attack on the outer REML/LAML derivative path —
the real wall-clock black hole (coord_corrections, not Newton/PIRLS):
- **BLAS-3 rigid Hessian fires for operator-backed designs.** The cycle-0
  `hessian_qp` (~8s) and directional `gradient_reload` (~8s) floors were the
  BLAS-3 override bailing to the per-row BLAS-1 SYR scatter whenever the
  marginal/logslope design is operator-backed (always, at biobank scale, via the
  #461 influence absorber / #978 overlap-Z). The override now chunks via
  `try_row_chunk` and fires for any non-sparse design (sparse still routes to the
  sparse-aware scatter); a `[STAGE]` line logs why the fast path was/wasn't taken.
- **Jeffreys H_phi drift base hoisted** out of the per-direction loop — the
  β-fixed part of the coord_corrections H_phi correction is computed once instead
  of re-streamed per smoothing direction (full batched single-pass contraction
  landing on top).

## gamfit 0.1.212 (2026-06-15)

First publishable build since 0.1.209 — the 0.1.210/0.1.211 wheels failed the
build.rs ban gate (agent-collision leftovers) and never reached PyPI. This
ships everything since 0.1.209 with the gate cleared (USE-or-DELETE, no `_`
silencers): the row-kernel dense/directional override defaults now run the
extracted generic per-row path (genuinely consuming their args) while the rigid
kernel overrides with the BLAS-3 fast path; heartbeat scope guards bound + dropped
explicitly; dead `is_ext` removed. Folds in the real BMS perf work:
**BLAS-3 Jeffreys hessian_qp** floor, **BLAS-3 gradient_reload** floor, the
**same-β rigid third/fourth-tensor cache** (the genuine coord_corrections
collapse — the rigid path was rebuilding the per-row tensor every outer eval),
canonicalise fast-path, seed-screen skip on warm hits, warm-start BFGS metric,
and heartbeat CPU/active-scope diagnostics. Correctness: #740 ψ outer-gradient
KKT correction + the outer-HVP cross-ρψ second-order sign and full-H⁻¹ β̈ solve
(audit-confirmed), Murphy-Topel oracle fixture redesign.

## gamfit 0.1.211 (2026-06-15)

BMS biobank-fit perf + diagnostics. (1) **BLAS-3 rigid dense joint-Hessian**
for the post-step Jeffreys/Firth KKT-residual term (the `gradient_reload`
~8s/cycle floor) — same BLAS-1→BLAS-3 chunked-Gram treatment as the hessian_qp
fix, bit-for-bit, full-data dense-design gated (subsample/HT fall through).
(2) **Named heartbeat scopes for rigid coord_corrections** so the process
monitor localizes where that step's time goes. (3) ψ outer gradient routed
through the unified KKT-residual correction when the inner residual r≠0 (#740),
vanishing at exact KKT. NOTE: the BLAS-3 inner-kernel fixes (this + 0.1.210's
hessian_qp + 0.1.208's coord_corrections) all gate on the full-data
unit-weight dense-design path; a run is the truth test for whether the biobank
fit hits that gate (0.1.208's coord_corrections fix did not visibly land, under
investigation).

## gamfit 0.1.210 (2026-06-15)

BMS biobank-fit perf: **BLAS-3 rigid joint-Hessian directional derivative**
(the Jeffreys/Firth head term that dominated the inner `hessian_qp` floor). The
rigid (`flex=false`) path previously scattered the per-row contracted third
tensor into the dense p×p Hessian via per-row rank-1 SYRs — O(k·n·p²) BLAS-1
that never reached a BLAS-3 kernel (the same root cause as the historical
"55s Jeffreys spike"). It now projects each row-chunk with GEMMs and closes with
`Xᵀdiag(w)X` per chunk (k·n/chunk BLAS-3 calls), reading the per-row third tensor
from the shared cache once instead of per Jeffreys column. Bit-for-bit (only
summation order changes); claims only the full-data unit-weight dense-design
case, subsample/HT-weighted paths fall through unchanged. The ~8s-per-Newton-cycle
`hessian_qp` contribution drops to well under 1s — and it recurs on every inner
cycle of every inner solve, so it compounds across the whole fit. Also includes
the #740 KKT-residual correction for the ψ (ext) outer gradient (gradient side;
vanishes at exact KKT so converged fits are byte-unchanged).

## gamfit 0.1.209 (2026-06-15)

BMS biobank-fit perf + diagnostics bundle (no behavior change to the converged
fit). (1) **Warm-start seeds the outer BFGS iter-0 metric** (`1/‖g₀‖`) so the
first line-search step is accepted at α=1 instead of bracketing — kills the
~3 redundant full-inner-solve `Value` probes per warm fit (same certified ρ).
(2) **Skip the cold seed-screening cascade on a warm-start hit** — the validated
warm ρ goes straight to BFGS (saves ~43s/warm fold; cold fits still screen).
(3) **canonicalise skips the redundant post-T invariant double-RRQR when T is
identity** (the clean rank-full case) and uses a BLAS-3 MAP Gram (~12s→~2s/fit,
bit-identical). (4) **Heartbeat diagnostics**: the process-monitor line now
reports a true `cpu=N/M cores` utilization signal (from /proc), the
currently-active operation + its elapsed time, and a progress fraction — instead
of the misleading `active_threads=0`.

## gamfit 0.1.208 (2026-06-15)

BMS perf: **BLAS-3 chunked Gram for the rigid directional-Hessian drift**
(`coord_corrections`, the per-ρ REML logdet-gradient drift). The rigid
(`flex=false`, biobank) path previously folded each of n rows × k directions
with per-row rank-1 SYRs (BLAS-1, memory-bandwidth-bound); it now routes through
the same chunked `Xᵀdiag(w)X` GEMM structure the flex path uses — k GEMMs per
chunk instead of n·k rank-1 updates. Bit-identical drift (the rigid accumulator
has no h/w blocks; non-contiguous/subsample rows keep the per-row fallback). On
the biobank LOSO fold (n≈326k, k=8) this collapses the dominant `coord_corrections`
step from ~3.5 min to seconds. Also lands the redesigned Murphy-Topel oracle
fixture (#1028) and HVP ψ-gradient FD-attribution instrumentation.

## gamfit 0.1.207 (2026-06-15)

Generative-dispersion cluster (#1124 / #1125), finishing and shipping a fix the
prior run landed on main but never released or verified:

- **#1124 (Python path):** `Model.sample_replicates` /
  `posterior_predictive_check` drew Negative-Binomial replicate counts at the
  construction **seed** `theta = 1.0` instead of the estimated `theta_hat`, so
  replicate counts carried `Var = mu + mu^2` rather than `mu + mu^2/theta_hat`
  (far too overdispersed; wrong posterior-predictive p-values). The CLI
  `gam generate` path had been unified onto the canonical dispersion picker, but
  `gam-pyffi`'s `generative_replicates_impl` kept a *separate inline copy* whose
  NB arm read the seed. Routed it through the single
  `gam::generative::family_noise_parameter`, so the CLI and Python front-ends can
  never diverge on dispersion handling again.
- **#1125:** verified the per-row precision channel `exp(eta_d(x))` is threaded
  into `gam generate` for every dispersion location-scale family
  (Gamma/NB/Beta/Tweedie) and restored the missing regression coverage.

Verification: new deterministic cross-family `predict↔generate` per-row variance
agreement test (max rel dev ~2e-16 across all four families, including the
Tweedie φ = 1/precision reciprocal); restored Gamma-LS end-to-end CV test
(impliedK 9.5→0.9 across x); new Python `sample_replicates` NB regression
(recovers theta_hat, not the seed); all existing generative + #1057 replicate
suites green.

## gamfit 0.1.206 (2026-06-15)

BMS perf: **same-β reuse of the per-row cell-moment exact-cache**. Repeated
evaluations at a bit-identical coefficient vector β — the outer BFGS
`Value`→`ValueAndGradient` pair at one ρ, line-search re-probes, warm-start
replay — now reuse a fingerprinted, bounded (capacity-2, FIFO) exact-cache
instead of rebuilding the O(n·cells) quadrature from scratch. Reuse is gated on
exact byte-equality of every build input, so a hit is bit-identical to a rebuild
(gradient, Hessian, LAML cost unchanged). Also includes a scale-relative
penalized-deviance floor restoring small-response REML equivariance (#1127) and
a hermetic `.git/index`-based build audit (no `git` shell-out in maturin Docker).

## gamfit 0.1.205 (2026-06-15)

Performance bundle for the BMS biobank fit, no behavior change (all paths stay
bit-exact / KKT- and REML-certified). (1) **Skip redundant continuation
pre-warm on a warm-start cache hit** — when a fit seeds ρ/β from a structurally
matching parent (LOSO folds, multi-disease runs), the cold continuation
pre-warm seed (~160s/fit in the biobank log) is no longer recomputed. (2)
**Reuse the exact-Newton workspace across rejected inner trust-region cycles**
instead of rebuilding it. (3) **Parallelize the HVP host-pin per-row direction
fill** that previously ran serial. (4) **Parallelize the flat identifiability
audit** (per-block QR + per-column geometry). (5) **Cache the same-ρ assembled
outer Hessian operator** so the ~14-19s spectral factorization is not rebuilt
on the Value→ValueAndGradient pair (and line-search re-probes) the outer BFGS
issues at identical ρ. RAM headroom on the biobank box (~10 GB of 87 GB used)
makes the operator cache a safe memory-for-speed trade.

## gamfit 0.1.204 (2026-06-15)

Warm-start: cross-fit **β** transfer (Phase 2). A related later fit (notably an
LOSO fold whose reduced coefficient width differs, e.g. 37 vs 35) now seeds its
β from a structurally-matching parent fit's converged coefficients via
function-space projection through the gauge (`θ_new = (TᵀT+εI)⁻¹Tᵀ β_raw`), not
just ρ — the prior `[CACHE] beta-warm action=skip … length mismatch` becomes
`action=projected`. Exactness-preserving (the inner Newton + outer REML still
run to their KKT/REML certificate) and finite-guarded: any anomaly falls back to
cold β for that block, never erroring a fit. Also folds in a DSL fix routing
mgcv `bs='cr'/'cs'` through the 1-D B-spline dispatch.

## v0.3.117 — gam 0.3.117 / gamfit 0.1.203 (2026-06-15)

crates.io catch-up release: publishes to the `gam` crate all the engine work
that has already shipped to the `gamfit` wheel since gam 0.3.116 (gamfit
0.1.199). No new code lands here — this only bumps the crate version and tags
it so crates.io consumers of `gam` get the accumulated 0.1.200–0.1.203 work.
Highlights of what is now on crates.io:

- Batched row-parallel `coord_corrections` in the outer REML/LAML gradient
  (gamfit 0.1.203, #979): biobank-scale gradient evals no longer idle ~80–95s
  per outer iteration running k single-direction passes serially.
- Phase 0+1 cross-fit warm-start foundation (gamfit 0.1.203): descriptor-indexed
  `FitArtifact` + structural ρ-transfer that seeds related fits (LOSO folds,
  row-population changes) from a prior converged fit's smoothing parameters.
  The non-test lib also no longer carried unconsumed Phase-2 scaffolding (the
  dead-code that would have failed `cargo build --lib` under the crate's
  `warnings = "deny"`).
- Noise-floor inner-Newton termination guard + certificate railed-coordinate
  false-positive fix (gamfit 0.1.202).
- BMS separation false-positive fix + parallel Jeffreys curvature + nested-BLAS
  pin (gamfit 0.1.201).

See the per-wheel entries below for the full detail of each item.

## gamfit 0.1.203 (2026-06-15)

Performance: the outer REML/LAML gradient evaluation no longer recomputes the
per-coordinate log-det drift corrections one direction at a time. At biobank
scale (n≈3e5, k=8 smoothing coordinates) that per-coordinate loop ran k full
n-row passes serially — leaving the machine idle (~80–95s per gradient eval,
repeated every outer iteration) because each thin single-direction crossproduct
could not fill the thread pool. The site now routes through the family's batched
`hessian_derivative_corrections_result` hook whenever it is advertised (the BMS
exact joint-Newton workspace fuses all k directions into ONE row-parallel pass
that amortizes the per-row cached cell-moment / third-tensor work), turning the
former `coord_corrections mode=serial(inner-parallel)` into
`mode=batched(row-parallel)`. Other families keep the existing
parallel/serial single-direction fallback unchanged; results are identical
(same negation/sign semantics, solver still runs to its KKT/REML certificate).

Also lands the Phase 0+1 cross-fit warm-start foundation (descriptor-indexed
FitArtifact + structural ρ-transfer) used to seed related fits (LOSO folds,
row-population changes) from a prior converged fit's smoothing parameters.

## gamfit 0.1.202 (2026-06-15)

Robustness: bernoulli marginal-slope LOSO/biobank fits no longer spin to the
inner-Newton cycle cap, and a post-fit certificate false alarm is silenced.
- Noise-floor inner-Newton termination guard: when the trust region collapses
  and every line-search step is rejected at the ~1-ULP floor (the objective is
  flat along the gauge-flat coupled direction), the solve now TERMINATES and
  judges convergence on the identified (range) subspace instead of spinning to
  the 1200-cycle cap. The line-search early-exit reject also gained a rounding
  tolerance so a numerically-flat trial is not rejected on 1 ULP.
- Certificate (first-order optimality self-audit) railed-coordinate fix: the
  audit now projects its gradient-vs-value directional check onto the FREE
  (non-bound-active) coordinates, so a legitimate KKT optimum with a smoothing
  parameter at a box bound no longer reports a spurious GRADIENT-OBJECTIVE
  DESYNC. Diagnostic-only; no fitted result changes; still fires on genuine
  interior desyncs.

## gamfit 0.1.201 (2026-06-14)

Performance: biobank marginal-slope fits no longer stall on the per-coefficient
Jeffreys/Firth curvature. Two fixes to the joint-Newton hot path:
- The exact Jeffreys curvature term H_phi was built by a serial p-pass loop, each
  a full-data directional-derivative sweep (~55s deterministic on n~2e5, p=35,
  arming at the ill-conditioned converged cycle). The p independent directions
  now evaluate in parallel across the Rayon pool (bit-identical math).
- Nested faer-BLAS GEMMs inside Rayon row-parallel assembly were pinned to
  single-thread, collapsing rayon x BLAS thread oversubscription (~300 ->
  cores) with no caps and no environment variables. gam owns parallelism and
  saturates the hardware with one level of fan-out.

## gamfit 0.1.200 (2026-06-14)

PyPI Linux-wheel refresh with post-0.1.199 fixes for generation,
dispersion location-scale prediction, survival and shape-constrained fitting,
multinomial convergence, and SAE dictionary robustness.

Fixes:
- Negative-binomial `generate` and `sample_replicates` now use the fitted
  `theta_hat` instead of the seed/default theta.
- Gamma, NB, Beta, and Tweedie dispersion location-scale generation now threads
  the fitted per-row dispersion channel, fixing homoscedastic synthetic output
  from heteroscedastic fits.
- Dispersion location-scale fits now assemble covariance and EDF consistently,
  including the converged orthogonal path, and prediction gates cover Gamma, NB,
  and Tweedie cases.
- Posterior mean observation bands now use per-row `sigma(x)` for
  heteroscedastic location-scale families.
- Interval and transformation-survival fitting is more robust: trial-point
  non-convergence is treated as high cost, interval warm starts stay feasible,
  and constrained Newton steps respect the monotone cone.
- Multinomial and BMS convergence paths were tightened with residual-stall
  Newton-decrement certification, identified-subspace stationarity checks,
  reduced Schur preconditioning, and removal of hot-path diagnostic
  eigendecompositions.
- `average_derivative` and default `difference_smooth` behavior have regression
  fixes.

SAE / manifold:
- K>1 SAE dictionary fitting is more stable under deflation and rank reduction.
- JumpReLU active-set support is canonicalized.
- Curvature reports now expose delta-method curvature SE through the Python
  facade.
- Cylinder topology race and born-atom uncertainty-band work is included.

Build / packaging:
- Repairs test-target builds after the module split and dispersion type changes.
- Replaces mechanical split fragments with named modules across major Rust
  subsystems.
- Removes stale local workflow/scripts/examples and redundant stub README files.

## v0.3.116 — gam 0.3.116 / gamfit 0.1.199 (2026-06-14)

Large crate + wheel release rolling up the unreleased work since 0.3.115. The
headline is a wave of new user-facing modeling capability — interval-censored
survival from a formula, full multinomial prediction inference, expectile GAMs,
magic Poisson auto-detection, restricted mean survival time, exact full-conformal
prediction intervals, and per-term likelihood-ratio tests — alongside a deep
solver-robustness cleanup (the derivative-free compass-search optimizer is gone,
several silent-failure crutches now fail loud), the SAE/manifold interpretability
stack, and a broad reference-quality test expansion. Build, rustfmt, and the
clippy correctness/suspicious gate are all green.

New modeling capability:
- feat(#1108): interval-censored survival is now fittable from a user formula —
  a dedicated `SurvInterval(L, R, event)` response (disambiguated from
  delayed-entry `Surv(entry, exit, event)`) materializes the time basis at both
  boundaries with frozen-knot column reuse and routes through the latent-survival
  path. Completes the deferred wiring step; covered by a gauge-invariant σ̂
  truth-recovery + match-or-beat-lifelines test.
- feat(#1101): multinomial-logit inference completeness — H⁻¹ covariance,
  `MultinomialPredictor` delta-method prediction intervals + standard errors,
  posterior_predict, and a per-term Wald summary, wired through the FFI and the
  Python surface. The fitted spec is frozen so `predict` rebuilds the exact
  fitted design.
- feat(#1100): expectile GAMs via a LAWS (asymmetric-least-squares) outer loop.
- feat(#1065): magic-by-default Poisson auto-detection for non-negative integer
  count responses.
- feat(survival): restricted mean survival time (RMST) output.
- feat(#1054, #942): exact full-conformal prediction intervals —
  `predict(interval='conformal')` auto-routes to the exact Gaussian jackknife+
  path and the exact GLM full-conformal engine, surfaced to Python.
- feat(#1063): per-term likelihood-ratio statistics with a magic Lawley–Bartlett
  correction (dispersion-carrying known-scale jets for Gaussian/Gamma), through
  the FFI and Python API.
- feat(#1032): `FitResult::ResidualCascade` variant + `fit_from_formula`
  auto-route with a quasi-uniformity guard, plus serde state replay.
- feat(#1057): generative replicate sampling + posterior-predictive checks
  (`Model.sample_replicates`).
- feat(#1049): posterior_predict for the Bernoulli marginal-slope path.

Geometry, manifolds & SAE interpretability:
- feat(#1061): SPD / Grassmann / Stiefel / Poincaré are now selectable response
  geometries.
- feat(#1104, #944): `ConstantCurvature` response manifold with an end-to-end
  curvature-as-estimand inference layer (κ̂ + profile CI + κ=0 LR from a real
  fit).
- feat(#1097, #1099, #1102, #1103, #1055): per-atom Riesz-debiased functionals,
  per-atom curvature profile-likelihood CIs with flat-cusp handling,
  cross-checkpoint atom-trajectory dynamics, and any-n e-value atom-smooth
  significance — wired end-to-end into `dictionary_report` and the Python facade
  (`Model.debiased_functional()`, `sae_checkpoint_dynamics`).
- feat(#1026): evidence-scored curved+linear-tail hybrid dictionary split made
  load-bearing on reconstruction, with leave-one-atom-out held-out EV
  attribution.
- feat(#1058): anytime-valid structure certificate surfaced post-fit.
- feat(#1038): exact cross-row integration-by-parts Woodbury on the arrow
  evidence cache (wired at the assembly site).

Solver robustness (deslop):
- The derivative-free compass-search optimizer is removed entirely: the last
  gradient-free callers (Weibull / transformation survival baselines) were
  migrated to exact-gradient BFGS, and `Solver::CompassSearch` and its dispatch
  were ripped out — it could hang, so it is gone.
- The arrow/Schur Woodbury path now fails loud instead of propagating a silent
  NaN.
- Removed the wall-clock fake-convergence early-exit in blockwise PIRLS.
- Cost-stall convergence must now clear the gradient tolerance, not just a flat
  cost.
- Replaced the magic 16-iteration ridge-escalation cap with a principled
  Gershgorin spectral ridge.
- Stopped silently clamping negative-eigenvalue smoothing corrections into a
  fake-PSD covariance; refuse phantom zero penalty-logdet ρ-derivatives instead
  of silently desyncing the outer optimizer.
- Constrained joint-Newton path: Newton-decrement certificate wired in,
  negative-curvature reflection convexifies the QP (#1040).

Performance:
- perf(#1017): CPU-resident reduced-Schur operator for the SAE matvec — factored
  (Lᵢ, Yᵢ) residency replacing the dense p×p block, with a work-based offload
  predicate and a parallelized per-row matvec.
- perf(#1082): parallelize competing-risks CIF assembly over the (independent)
  row axis behind the rayon nesting guard + a row-count gate, byte-identical to
  the serial path; n-scaled outer-gradient floor across all families;
  warm-started β in the spatial outer loop; right-sized quality-test fixtures.

Reference-quality tests: a broad new wave across the mature comparators
(lifelines interval censoring, VGAM multinomial smooth-by-factor, mgcv tensor
products, frailty survival, PyMC/InterpretML, conformal coverage, and more).

Known limitations (tracked, not regressions): several quality fits still exceed
the wall-clock budget (#1082, #1116) and the survival marginal-slope path is slow
on some bases (#979) — under active investigation; these are perf/coverage gaps,
not correctness defects in the shipped surface.

## v0.3.115 — gam 0.3.115 / gamfit 0.1.198 (2026-06-13)

Crate + wheel release rolling up the unreleased work since 0.3.114: the O(n)
spline scan reaching the `fit_from_formula` library entry point and gaining
order-3 (quintic) support, a new exact O(n) multi-term additive backfitting
module, a Bessel-K accuracy fix in the Matérn/Duchon radial lattice, the exact
dense softmax-entropy row Hessian, the Murphy–Topel generated-regressor
correction, analytic Jacobi-field exp-map VJPs on curved manifolds, sphere-chart
isometry-defect certification, and the gam-pyffi module carve-out — plus a batch
of solver-robustness fixes and a broad new reference-quality test wave.

O(n) spline scan — library entry point + order-3 (#1030, #1044):
- feat: `fit_from_formula` now auto-routes a qualifying single 1-D Gaussian
  smooth through the exact O(n) diffuse-REML Kalman/RTS scan (new
  `FitResult::SplineScan` variant), matching the FFI/CLI auto-routing that
  already shipped; the conservative detector falls every other shape through to
  the dense path unchanged.
- feat: order-3 (quintic) smoothing splines (`λ∫(f‴)²`) via an exact diffuse
  leading-block smoother for the two partially-diffuse leading nodes, with
  symmetric (Jacobi) equilibration + iterative refinement of the 6×6
  leading-block solve so the κ≈δ^{-5} stiffness is resolved at heavy smoothing.
  Validated against the order-general dense exact-posterior oracle to 1e-6·SD
  (posterior) and 1e-7 (REML differences); the auto-route now covers
  penalty_order ∈ {1, 2, 3}.

Exact O(n) multi-term additive backfitting (#1034 item 3):
- feat: new `gam::solver::scan_backfit` — `fit_scan_backfit` /
  `fit_scan_backfit_at` solve `y = α + Σⱼ fⱼ(xⱼ)` with exact O(n) spline-scan
  inner smoothers, certified against a dense joint penalized solve, with
  per-term λ selection validated match-or-beat vs a dense joint-REML grid on
  truth recovery. (The module had been committed unwired and uncompiled; it is
  now wired, compiled, and covered by three oracle tests.)

Numerical accuracy:
- fix(basis): `K_{l+½}` Bessel coefficients now come from the exact upward
  recurrence instead of Lanczos-Γ at integer arguments, removing ~1 ULP errors
  that amplified in the near-cancellation `r^{-0.5}·K_{1/2} + r^{0.5}·K_{3/2}`
  radial-derivative sums of the Matérn/Duchon lattice.
- feat(#1035): exact Chebyshev jet tower for the radial-profile derivative
  channels.
- feat(#1038): the softmax assignment-entropy prior's exact dense per-row
  Hessian in logits is wired through value / log|H| / θ-adjoint / ρ-trace
  together, so the criterion and its gradients differentiate the operator the
  prior actually defines (previously only the diagonal was stored).

Geometry + manifolds:
- feat(#944): analytic Jacobi-field `exp_map_vjp` for κ ≠ 0 in constant-curvature
  geometry (replacing the κ = 0-only path), with a by-reference `dot` fix.
- feat(#1019): certified sphere-chart isometry-defect functional plus a post-fit
  measurement/logging pass.
- feat(#1026): per-atom fitted-turning Θ measurement for the EV-vs-Θ signal.

Other:
- feat(#1028): Murphy–Topel generated-regressor variance correction assembled
  with J_zeta accumulation (BMS).
- feat(#1017 Phase 0): driver-level parallel topology-candidate selection.
- feat(#1033): ψ-Gram tensor gradient coverage surfaced at the eval_full seam,
  with n-free k-space ψ-derivatives feeding the Gaussian gradient.
- fix(#1036): the cross-seed structural detector parses the real non-PD wording.
- fix(audit): RRQR keeps its own block when it already named the lower-priority
  alias side.
- refactor(#780): the gam-pyffi monolith is carved into focused modules
  (`python_literal`, `benchmark_scores`, `competing_risks_decode`,
  `summary_render`); restored a missing `PyValueError` import that would have
  failed the wheel build.
- perf(#1035): caller-thread line-search LL sweep / serial small row-folds; a
  measured-no-op row-fold gating was reverted (it implied a perf win that was
  not observed).
- gpu: dropped the dead Phase-3 Cornish–Fisher Pólya-Gamma oracle.
- tests: new reference-quality gates (grid_spline_2d vs mgcv te() #1031,
  multinomial softmax truth-recovery vs mgcv/VGAM #715, competing-risks
  total-probability identity #1025, survival marginal-slope convergence #1040,
  measure-jet aniso isotropic-fallback regression, thread-pool fit-invariance
  #1045).

## v0.3.114 — gam 0.3.114 / gamfit 0.1.197 (2026-06-13)

Crate + wheel release rolling up the O(n) spline-scan auto-routing, the
measure-jet simple/multiscale split and conditioned-frame ψ-Gram tensor, the
multinomial formula-fit corrections, the Matérn-anisotropy isotropy fixes, and
a large batch of formula-parsing / solver-robustness fixes. (On crates.io this
also rolls in the 0.3.113 content — `gam 0.3.113` was tagged in-tree but never
reached crates.io, which was still on 0.3.112; this release re-syncs both
registries.)

O(n) spline scan — auto-routing + saved models (#1030, #1034):
- feat: the FFI and CLI now auto-route a single 1-D Gaussian smooth through the
  O(n) spline scan end-to-end, so `s(x)`-only Gaussian fits skip the dense
  REML path. Includes an n=1e6 spline-scan benchmark + a biobank-scale
  no-regression certificate.
- feat: a lossless `SplineScanState` saved-model representation channel with a
  bit-for-bit JSON round-trip gate (serde_json `float_roundtrip`), validated
  through `FittedModel`; the scan-model payload exempts dense-only fields.
- feat: order m∈{1,2} routed through detection + callers (m=1 dense oracle),
  with order-general scan API names and a match-or-beat truth-recovery gate vs
  mgcv for the spline scan.

measure-jet — speed + accuracy (#1039, #1033b):
- perf: SIMPLE (single-scale) mode is the default again, fixing the ~12× slowdown
  vs Matérn (#1039); the simple/multiscale auto-mode split is documented and
  hardened, with a θ-invariant Gram cache for fixed-design ρ-only trials.
- feat: the certified Chebyshev-in-ψ Gram tensor is built in the conditioned
  frame and fires on real spatial smooths (#1033b); n-free Gaussian ψ-gradient
  from the tensor derivatives, with a finite-difference gradient certificate.
- fix: density-free default α = 3/2 (from the ρ^{3−2a} derivation).

Multinomial formula fits (#715):
- fix: standardize the unpenalized parametric columns in the multinomial formula
  fit; restore exact outer curvature for double-penalty multinomial formula fits;
  floor the multinomial inner KKT tolerance at the softmax f64 noise floor.
- tests re-grounded onto a like-for-like mgcv REML / `select=TRUE` comparator;
  no quality bar weakened.

Matérn anisotropy (#1042):
- fix: honor an explicit all-zero `aniso_log_scales` as isotropic on the Matérn
  forward path; gate the anisotropy auto-seed on center-strategy provenance
  rather than seeding unconditionally.

Survival marginal-slope (#1040, partial):
- fix: the joint-Newton inner solve now takes a scale-invariant relative
  objective-plateau exit on flat REML valleys, so survival marginal-slope fits
  terminate instead of grinding to the inner-cycle ceiling every outer ρ-eval.
- fix: early-certificate inner exits report the residual they actually certified
  on, so the terminal verdict can no longer print `converged=true … residual=inf`
  (inner-report truthfulness).

Other fixes:
- fix(#1025): route uncoupled multi-block inner solves to the exact
  block-coordinate path; joint competing-risks transformation truth-recovery.
- feat(#939): wire the Lawley/Bartlett cumulant summary fields through `main`.
- fix(#1036): generic cross-seed structural-failure bail in the seed cascade.
- fix: accept ragged `Column` lists in `write_columns_csv` by NA-padding the tail
  (unblocks the varying-ν Matérn / sphere-geodesic / Grassmann quality suites).
- fix: peel mgcv `c(...)`/`(...)` R-vector wrappers in `split_list_option`, so
  `te(x,y, bs=c('tp','tp'), k=c(5,5))`-style per-margin options parse correctly.
- fix: width-guard the cached marginal/logslope β-hints before block assembly so
  a stale hint falls back to a clean cold start instead of tripping the
  block-spec contract.
- perf: parallelized SAE row assembly, iterator/slice-dot REML and SAE reductions,
  fused ρ-posterior cost+gradient eval, and gradient-only routing for
  high-dimensional REML ρ fits.

## v0.3.112 — gam 0.3.112 / gamfit 0.1.195 (2026-06-11)

Crate + wheel release rolling up the over-dispersed-Gamma predictive-interval
fix, the random-slope / factor-smooth predictive-quality cluster, and the
marginal-slope hang / OOM fixes that unblock the survival + binary
`marginal-slope` study.

Predictive intervals — over-dispersed Gamma / Tweedie (#1018):
- fix: `gamma_quantile` now inverts an *owned* regularized lower-incomplete-gamma
  CDF (`regularized_lower_gamma`, the Numerical Recipes power-series / Lentz
  continued-fraction split with the leading factor kept in logs) instead of
  `statrs::gamma_lr`, which hard-clamps to `0` for every `x ≤ 1.11e-15`. In the
  small-shape lower tail (`a ≲ 0.1`, the moment-matched `k = μ²/V` of a strongly
  over-dispersed predictive) the clamp zeroed the Halley residual and walked the
  iterate *up* to `~1.6e-15` — so a nominal 2.5% bound carried up to ~19% of the
  mass and the interval under-covered on the low side. Round-trip
  `P(a, gamma_quantile(p,a,1)) ≈ p` now holds to 1e-9 down to `q ~ 5e-33`, pinned
  by an independent self-contained oracle.

Random-slope / factor-smooth predictive quality vs lme4 / mgcv (#903):
- fix: `bs="re"` is now the parametric random intercept+slope `[1, x]` mgcv's
  `(1 + x | g)` denotes, not a piecewise-linear B-spline under the pooled-knot
  heuristic (~6 wiggly coefs/group). The over-parameterized term ill-conditioned
  the REML/joint-Newton solve (minute-long fits) and broke partial pooling;
  group slopes now shrink toward the fixed population trend (7 s fit, 2 coef/group
  edf), beating no-pooling OLS out of sample.
- fix: cap the `bs="fs"` shared marginal (and `bs="sz"`) at mgcv's default
  `k ≈ 10`; the pooled heuristic gave ~24 functions/group and REML over-fit the
  shared shape. fs recovers at 0.0528 (beating mgcv's 0.0548).
- tests re-grounded onto the correct comparator model (random slope vs lme4's
  `(1+x|g)`, not the shrink-to-zero `fs`); no quality bar weakened.

Marginal-slope (binary + survival) — hang / OOM (#979, partial):
- fix: the inner joint-Newton no longer hangs to its 1200-cycle ceiling on
  fully-rejected cycles (β reverts and an interior block pins `max(block_radii)`,
  so every subsequent cycle was bytewise identical and rejected for the same
  reason — ~120 s burned per outer ρ-evaluation, the survival "hang").
- fix(large-scale): the survival SMGS phase-4b observability / rank-diagnostic
  blocks densified operator-backed designs unconditionally and OOM-killed the
  host at n=320k; they now go through a pre-allocation byte budget
  (`try_to_dense_by_chunks_budgeted`, 256 MiB/matrix) and `warn!`-skip on
  refusal, while real numerical-rank failures still propagate.
- correctness: the Jeffreys curvature in the coupled-joint outer LAML is now the
  exact Daleckii–Krein form (dropping the `K²` vec-Gram surrogate that put ~1e20
  phantom curvature on floored eigenpairs and froze the inner step along
  Firth-active directions), with its exact divided-difference drift, the Tier-B
  `Φ(β̂)` value folded back into the outer cost, and PSD projection on
  SPD-requiring step paths. The deeper inner-Newton convergence on the hardest
  binomial-location-scale Firth-active fixtures remains in flight; #979 stays
  open for that.

Build / CI:
- #901: keep the non-Gaussian value path on the spectral LAML.
- CI: per-binary test wall-clock caps now use GNU `timeout` (process-group kill
  by default) in place of the hand-rolled `setsid` watchdog; removed the stale
  precheck / hillclimb scripts.

## v0.3.111 — gam 0.3.111 / gamfit 0.1.193 (2026-06-10)

Crate + wheel release rolling up the factor-smooth predictive-quality fix and
the SAE-manifold streaming/LLM-scale restoration on top of the green tree.

Factor-smooth / random-slope quality (#903):
- fix(fs): the `bs="fs"` factor smooth now penalizes **each null-space
  dimension separately** (one rank-1 `I_L ⊗ z_k z_kᵀ` per null direction, each
  its own shared smoothing parameter), mirroring mgcv's
  `smooth.construct.fs.smooth.spec`, and drops the non-mgcv range ridge. The
  prior single *combined* null penalty (one λ for intercept + slope) could not
  express the distinct random-intercept vs random-slope variances, so
  per-group slopes got no partial pooling and the held-out per-subject forecast
  inherited full no-pooling variance. REML now fits both variances, tracking
  lme4's correlated-RE BLUP.

SAE-manifold streaming (LLM-scale fitting):
- restore + wire the out-of-core streaming joint-fit driver
  `run_joint_fit_arrow_schur_streaming` (re-seeds each chunk via `chunk_init`,
  never materializes the `(N×M)`/`(N×K)` per-row buffers) plus an in-memory
  entry `fit_streaming_in_memory`, with a chunk-size-invariance contract test.
  This is the memory-bounded fit path for the LLM-scale teacher; the in-core
  driver cannot scale to N = billions of tokens.
- feat(#972/#977): closed-form streaming polar frame refresh
  (`refresh_active_frames_from_data`) — the U-block of the alternating
  block-coordinate ascent that complements the border C-block Newton step.

Build / correctness:
- bms #905 conditional `E[z|C]`/`Var(z|C)` Auto gate; #740 contracted ψψ
  second-order hook; #1000 centered Gaussian outer-REML λ-search; assorted
  ban-gate and unused-symbol cleanups to keep `cargo check --tests` green.

## v0.3.110 — gam 0.3.110 / gamfit 0.1.192 (2026-06-10)

Crate + wheel release rolling up the SAE-manifold identifiability-certificate
work (#980/#981/#907/#995/#996/#998) plus the build/lint fixes that unblock a
green `cargo check --tests` + Rustfmt across the workspace.

Identifiability / SAE-manifold certificate:
- feat(#998): the residual-gauge certificate now realises within-atom gauge
  orbits **exactly** in the model's own (decoder, coordinate) parameter space.
  The coordinate-motion field comes from the group action (circle/torus
  shifts, flat-patch so(d) rotations); the decoder compensation is profiled
  out by least squares; the leftover residual is the orbit's true data cost —
  exactly zero for a basis closed under the action, positive otherwise, so
  **basis closure is computed, not declared**. All pinning of true model-class
  symmetries flows through the injectable `OrbitPenaltyOperator` channel.
  `residual_gauge_exact` merges exact within-atom verdicts with the calibrated
  frame path for spheres / unviewed atoms / cross-atom families.
- fix(#995): the per-generator verdict uses the relative curvature fraction
  `‖R ξ̂‖² / σ_max(R)²` (kept magnitudes, survives a full-rank pinning span)
  calibrated by a computed mean-frame `lowering_error` scale, so compression
  artifacts are never read as a pin. Shipped per-generator through the FFI.
- feat(#996): the discrete-mixture rung refines locally around the coarse
  `MIXTURE_K_LADDER` winner until bracketed, so off-ladder truths (k = 4/6/8)
  are named exactly instead of snapping to the nearest rung.
- test(#980/#981/#907): Theorem-2 arm, the circle-read-discretely two-verdict
  race, and the repeated-draw calibration sweep.

Build / lint:
- fix: drop a dead `policy` field (gamlss dispersion family), wire the new
  `DispersionLocationScale` `FitRequest`/`FitResult` arms + `ReportInput`
  convergence fields through the Python FFI, annotate ambiguous-float arrays
  in the #974 residual-factor test, and `cargo fmt --all`.

## v0.3.109 — gam 0.3.109 / gamfit 0.1.190 (2026-06-10)

Crate + wheel release rolling up the correctness fixes and the new
inference / SAE / topology / survival surface landed since gam 0.3.108 /
gamfit 0.1.189. (The intervening gamfit 0.1.190 wheel never reached PyPI —
its build broke on a `gam-pyffi` site that lagged the #983 `theta_fixed`
refactor; that and the rest of the `-D warnings` / ban-gate breaks are fixed
here, so the wheel builds green again.)

Family / link / survival correctness:
- fix(#947): binomial 4th central moment uses the 3rd inverse-link pdf
  derivative (μ''''), not the 5th.
- fix(#948): exact binomial derivative towers in the saturated tails — no
  clamped-μ surrogate (the derivative path is the derivative of the evaluated
  row loss).
- fix(#953): integrated observation variance uses the Tweedie φ / Gamma
  shape, not a hard-wired φ=1.
- fix(#961): the link is validated against the explicit family instead of
  inferring a conflicting family from the link.
- fix(#963): exact public Log inverse-link jet on the predict surface (the
  solver keeps its internal clamp).
- fix(#964/#965/#966): survival FFI fallback S→H, negative-time handling, and
  hazard-differencing math corrected.
- fix(#983): a fixed `--negative-binomial-theta` is honored end-to-end
  (`theta_fixed` routed through every call site) instead of being silently
  re-estimated.

Geometry / manifolds / REML:
- fix(#949): Sphere / Euclidean `sectional_curvature` validates a
  nondegenerate tangent 2-plane before dividing by the wedge area.
- fix(#950): `simplex_exp_map` CLR requires a strictly-positive base.
- fix(#951): `signed_log_sum_exp` propagates +∞ log-magnitudes correctly.
- fix(#952): Fisher-Rao precision blocks validated PSD (PD on the Cholesky
  path), not merely symmetric with non-negative diagonal.
- fix(#954): optimizer stationarity uses the shift-invariant
  ‖grad_k‖/‖grad_0‖ measure at both call sites.
- fix(#955): the Euclidean differential is raised to the Riemannian gradient
  through the metric.
- fix(#957): the trust-region radius is constrained 0 < radius ≤ max_radius
  before the first step.
- fix(#967): a shared per-smooth λ makes the response-geometry tangent fit
  frame-equivariant.
- fix(#901): intrinsic pseudo-logdet over range(H_pen); the custom-family
  joint trace kernel uses the full spectral M⁺; the GLM cubic-correction drift
  stays in operator form (no near-null dense-C[v] roundoff blow-up).
- fix(#902): Matérn ψ-derivative penalty enumeration aligned with the forward
  gate (RKHS rule j ≤ ν + d/2).
- fix(#978): persisted + replayed global-orthogonality chart so an overlapping
  global + factor smooth on the same covariate residualizes consistently.
- fix(#780): the LinearOperator seam is wired into weighted_design_products.
- remove: the standalone SINDy sparse-dynamics module (STLSQ + library + FD +
  equation renderer, `SindyAtoms`) and its pysindy reference-quality suite.
  It shared no machinery with gam's REML/penalized-spline core, contradicted
  the REML-always selection policy (BIC + hard-threshold), and existed only as
  a pysindy benchmark target; its sole prospective consumer (#908 Manifold-
  SINDy) was unbuilt research. Removed with #482/#908/#945/#958/#959.

New surface:
- feat(#986): per-atom decoupled Extended Fellner–Schall as the primary outer
  at frontier ρ-scale (auto-switched into `run_outer`), with a matrix-free
  θ-HVP (#740) for the shared-border coupled correction.
- feat(#931/#935): the profiled criterion calculus — LAML as
  self-differentiating atoms over one sensitivity operator (factored H⁺;
  β̇ / ALO / influence / case-deletion / θ-HVP as its contractions).
- feat(#932): Taylor-jet tower algebra (`Tower4<K>`) — write a family's row
  log-likelihood once, derive its whole `RowKernel` derivative tower exactly.
- feat(#942): exact full-conformal prediction for penalized GAMs (Layer 1).
- feat(#944): a `ConstantCurvature` manifold family through κ=0 with exact
  κ-jets riding the #932 tower.
- feat(#973): streaming out-of-core border-Gram accumulator on a deterministic
  pairwise-reduction tree (order-invariant, resumable) + the streaming SAE
  corpus driver.
- feat(#974): structured-residual estimator with a single likelihood-whitening
  seam.
- feat(#972/#985/#987): low-rank Grassmann decoder frames; a sublinear
  candidate-atom index for active-set proposal; frontier distribution +
  two-tier Fisher-on-subsample harvest economics.
- feat(inference): SAE-manifold diagnostics — two-lens per-atom (presence vs
  behavioral coupling) diagnostic, residual-gauge certificate, Fisher-mass
  enrichment ordering, provenance-carrying RowMetric; plus the steering
  primitive (`sae_steer_delta` FFI + `ManifoldSAE.steer`) with output
  dosimetry, validity radius, and an off-manifold guard.
- feat(#907): discrete-mixture and union-candidate rungs in the topology race
  with selection-time stacking.
- feat(#741): `SmgsLiftViaT::lift_covariance_via_t` inference pushforward.

## gamfit 0.1.189 (2026-06-09)

PyPI wheel release on top of gam 0.3.108, rolling up the survival + predict
correctness work landed since gamfit 0.1.188:

Survival (transformation / location-scale AFT):
- fix(#892): reduced parametric-AFT time-warp gauge is now identified and
  verified end-to-end. The fit routes σ-scaling through the location channel
  (`η_t → η_t − log t`, warp `h ≡ 0`, so `u = (log t − μ)/σ`) and the predict
  path mirrors it — a 3-bug chain fixed forward: regime detection by all-zero
  `beta_time` (not `is_empty`, since finalize emits a non-empty length-`p` zero
  vector), keep the full-width zero-β time basis (warp nulls via β=0) so the
  scale-deviation primary keeps its full column count and the hazard dim guard
  holds. New e2e CLI fit→save→load→predict test tracks the lognormal truth.
- fix(#899): saved Weibull baseline scale recovered from the time anchor, not a
  stale/unidentified `exp(−β[0]/shape)`.
- fix(#900): factor-by-level smooths centered against their gated level
  indicator, so each `s(x, by=g)` group keeps its own per-group baseline.

Multinomial / GAMLSS:
- fix(#715): multinomial REML adapter skips the outer seed-screening cascade on
  the LM-damped formula path so a valid REML seed survives the canonical-gauge
  null direction.
- refactor(#780): extracted the binomial neg-log-q derivative math and the
  location-scale validators out of the gamlss.rs mega-file into
  `gamlss/binomial_q_derivs.rs` + `gamlss/validation.rs` (pure, no behavior
  change).

Predict guard tests:
- Added e2e truth-recovery regressions for tensor `te(x,z)` and by-factor
  `s(x, by=g)` predict on fresh grids (both paths verified correct).

## v0.3.108 — gam 0.3.108 / gamfit 0.1.188 (2026-06-09)

Crate + wheel release rolling up everything since gam 0.3.107. crates.io is at
gam 0.3.107 and PyPI has been stuck at gamfit 0.1.180 — the intervening
gamfit 0.1.181–0.1.187 wheels never published because the wheel build broke on
lib compile errors (`-D warnings`) that the per-push CI does not catch (Rust CI
runs nightly only). Those breaks are fixed here, so the wheels build green again
for the first time since 0.1.180. The `gam` 0.3.108 crate published to
crates.io; the gamfit 0.1.188 PyPI upload is separately blocked on the project's
10 GB storage quota (the wheels build but `twine` is rejected) — see #894.

New:
- feat(report): terminal smooth visualizer. `gam report` now prints a unicode
  block-glyph sparkline (`▁▂▃▄▅▆▇█`) of each smooth term's fitted partial effect,
  labelled with its x- and y-ranges — instant shape diagnostics with no plotting
  backend. Pure, read-only renderer in the new `gam::sparkline` module; faithful
  min→bottom / max→top mapping with graceful constant/NaN/empty handling.

Location-scale (GAMLSS) correctness:
- fix(#884): Gaussian location-scale models now persist and reconstruct the
  actual `response_scale`, and the noise σ-floor is response-scale equivariant —
  rescaling the response rescales σ̂ exactly instead of silently clamping at a
  fixed floor. Wired through the predictor, the CLI save path, and the gamfit
  FFI payload.
- fix(#684): Gaussian loc-scale uses a Gaussian (not GLM-upward-biased) seed risk
  profile, and the matrix-free workspace mean↔scale cross-block is the Fisher
  zero, not the observed `2κm` term.
- fix(gamlss): the Gaussian wiggle mean↔scale Fisher cross blocks are exactly
  zero (static, 1st- and 2nd-directional, and the μ↔logσ / logσ↔wiggle mixed
  β·ψ crosses), removing a spurious coupling in the joint Hessian.
- fix(#826): true dogleg globalization + scale-invariant Marquardt damping for
  the coupled location-scale inner Newton, which previously froze on tightly
  coupled problems.

REML / LAML:
- fix(#877): Gaussian REML is weight-scale invariant — λ̂ now scales exactly with
  weight magnitude via complete normalization and a weight-anchored seed +
  ρ-prior, so multiplying all weights by a constant no longer moves the fit.
- fix(#715): structural EDF floor uses true generalized eigenvalues; multinomial
  inner-Newton budget is decoupled from the outer `max_iter`/`tol`; Firth-bounded
  finite-saturated multinomial formula fits are accepted instead of mis-rejected
  as separation.
- fix(reml): custom-family LAML `log|S_λ|₊` value is synced to the gradient's
  pseudo-logdet classifier; the InnerAssembly TK correction value and gradient
  are guarded together (single `include_logdet_h`) so they cannot desync.
- fix(#808, #854): exact floored-pseudo-inverse and second-directional Hessian
  derivatives for the joint-Jeffreys spatial-adaptive family.

Smooths / latent / shape constraints:
- fix(#876): periodic latent Duchon decoder recovers a circle/torus instead of
  collapsing (spectral seed + seam-consistent jet that differentiates the
  periodic forward).
- fix(#873): shape-constrained smooths get a strictly-interior cold-start seed,
  equality-pair-safe interior projection, scale-invariant complementarity, and
  outer-KKT-gated soft acceptance.
- fix(#879): honest latent-fit diagnostics (projected gradient + reconstruction
  quality, scale-aware stationarity for the profiled-scale objective).
- fix(#880): hardened `duchon_function_norm_penalty` wrapper with regression
  tests.
- fix(#691): survival monotone-baseline I-spline — keep the convergent
  increment-space penalty (the value-space `Lᵀ S_B L` penalty remains the
  documented limitation pending survival-inner-solve work).

Prediction / diagnostics:
- fix(predict): prediction (observation) intervals now fold in estimation
  variance, not just the noise term.
- fix(survival-predict): a 2-arg `Surv(time, event)` predict builds a default
  time grid instead of erroring.
- fix(#881, #882, #883): post-fit diagnostic loaders carry the response and
  offset through, so diagnostics on reloaded models are correct.
- fix(alo): approximate-LOO standard errors use the estimated dispersion on both
  the geometry path and the diagnose refit route.
- fix(firth): link-general single-eta PIRLS Firth score shift.
- feat(#738, #741): the joint-Hessian path selects its representation by intent
  (matrix-free HVP for the inner solve, dense for the log-determinant) and
  exposes the full row-Hessian quotient as a `CompiledMap`.

Build / release hygiene:
- Restored workspace compilation: derived `PartialEq` on the
  `PhiScaledCovariance` / `UnscaledPrecision` covariance newtypes (the
  `array_values_equal` → native `==` cutover), removed a dead combined
  `GuardedCorrection::apply`, repointed the relocated `families::wiggle` helpers
  in the integration tests, and reformatted post-relocation lines. The release
  also rolls up the unpublished gamfit 0.1.181–0.1.187 changes.

## v0.3.107 — gam 0.3.107 / gamfit 0.1.186 (2026-06-08)

Crate + wheel release. crates.io was last published at gam 0.3.105 and PyPI at
gamfit 0.1.180; the intervening gam 0.3.106 / gamfit 0.1.185 tag failed to
publish (its top changelog entry did not name the tag, so the publish gate
rejected it), so this rolls up everything since — including the gamfit 0.1.185
entry below — into one good release.

Survival / monotone baseline:
- fix(#691): monotone-baseline survival fits converge again. A value-space
  I-spline curvature penalty (`Lᵀ S_B L`) was the principled fix for the
  monotone tail bias, but it enlarges the penalty nullspace and the survival
  inner PIRLS does not yet pin the likelihood-identified linear-trend direction
  over it: the penalized Hessian stays full-rank (no near-null space to project
  away) yet the constrained stationarity residual sticks at ‖g‖≈0.5 and the fit
  hits MaxIterations — a hard `IntegrationFailed`. The accompanying "range(H)
  stationarity rescue" rested on a misdiagnosis (it can never fire when H is
  full-rank) and is removed. Restored the converging increment-space penalty;
  the tail-bias trade-off is the documented #691 limitation, and the value-space
  penalty remains the real fix pending survival-inner-solve work.

Separation / Firth / inner solve:
- fix(#729, #715, #826): correct the Jeffreys/Firth merit sign and fold it into
  the coupled inner trust-region model so the baseline matches the trial and the
  K-block converges; scale the inner KKT tolerance by Firth score magnitude.
- perf(#729, #826, #808): O(p²) Gershgorin stabilizing shift, BLAS-3 assembly of
  the joint-Jeffreys curvature, and a cross-cycle cache keyed on beta.

Smooths / bases:
- perf(#813): dimension-aware tensor margin `k` to stop the ∏k product blowup,
  with a regression test.
- fix(#784): block-local sampled correction is smooth in rho (lambda-independent
  MC seed).
- fix(#787, #860): freeze the matern double-penalty nullspace-shrinkage decision
  across kappa rebuilds.
- fix(#854): exact second directional Hessian derivative for the spatial-adaptive
  family.
- fix(#780): commit the cyclic basis seam module.

Build / release hygiene:
- fix(#871): gate dead-code removal on gam-pyffi cross-crate reachability;
  restore modules the published wheel imports that looked dead to the `gam`
  crate alone.
- Unblock the workspace compile (stale bench/pyffi APIs), clear the rustfmt
  drift across the tree, and drop a dead lint-flagged rebinding.

## 0.1.185

- Revert matern double_penalty=false default regression (#787).
- Family-gated self-vanishing-mu cond-damping for marginal-slope inner solves (#787/#808).
- Operating-point warm-start of survival logslope initial_beta (g=0 seed-trap escape) (#808/#814).
- Suppress preconditioned-descent substitution at the joint-Newton step floor (#787 c12).

## v0.3.105 — gam 0.3.105 / gamfit 0.1.183 (2026-06-07)

Catch-up crate release. crates.io was last published at gam 0.3.103; this rolls
up every engine fix that shipped through gamfit 0.1.178–0.1.183 since then.

Survival marginal-slope (#808 family):
- fix(#808): reject only *channel-deleting* rawstack reductions. On clustered-PC designs the raw marginal / log-slope columns collide and the `[Time, Marginal, Logslope]` cross-block carry would zero the entire log-slope block — deleting the slope channel the model exists to estimate and diverging the inner solve. Non-destructive partial reductions are kept; any map that collapses a required channel to zero width is rejected and the unreduced design is used, leaving the near-null direction to Jeffreys conditioning. Guarded by a unit-tested predicate.
- fix(#834): continuation prewarm is now objective-opt-in, and the continuation seed dispatch / RE-Hessian guard / never-fail escalation reachability are pinned (#819/#737/#834/#860).

REML / ALO performance and correctness:
- fix(#862): ALO robustness weights are scoped to the owning `RemlState` instead of a process-global pointer-keyed map. Cold model sweeps reallocate a state at the same address with the same `n`; the old map then reused another formula's frozen ALO weights and recreated the 30–70× outer-REML grind on `smooth + a few linear covariates`. The frozen nuisance now lives on the surface and invalidates with the design in `reset_surface`.
- fix(#818): one comparable REML score across the Python APIs, so model-selection numbers line up between the formula and builder paths.
- fix(#819): materialize the sparse exact inner Hessian — `group()`-panel sparse-exact REML smoothing-correction no longer aborts.
- fix(custom/REML #824,#825,#837,#826): Firth consistency, Fisher block contract, joint-Newton rank floor, and stacked-solver η honored in custom-family assembly.

Smooths / bases:
- fix(#787): the Matérn formula defaults to `double_penalty=false`. The strictly positive-definite Matérn kernel has no structural polynomial nullspace, so the double-penalty ridge was spurious and flipped the learned-penalty count across the κ optimizer's design rebuilds ("joint hyper rho dimension mismatch"). An explicit `double_penalty=true` is still honored; Duchon/thin-plate keeps its native nullspace shrinkage (#754).
- fix(smooth/basis #822,#823,#851,#858): isotropic-Matérn κ axis, design storage / boundary handling, Duchon adaptive caches.

Predictions:
- refactor(#817): the moment-matched Gamma predictive interval is lifted into a pure, unit-tested `probability::gamma_moment_matched_interval` (exact conditional-Gamma limit, right-skew asymmetry, estimation-uncertainty widening, degenerate-input fallback). No behavior change to the #817 fix itself.

Diagnostics / families / linalg:
- fix(#864): `diagnose` keeps the response column instead of dropping it and aborting on its own training data.
- fix(#861): accept the redundant marginal-slope Firth flag.
- fix(#845,#846,#848,#849,#852,#855,#856): arrow-Schur / linalg / survival / sinkhorn correctness.
- fix(GPU/BMS #829,#831,#833,#835,#836,#838): trace SE, survival_flex integrand, 4th-order term, saddlepoint κ4, Mills seam.
- fix(SAE #841–#857): manifold correctness and streaming-logdet convergence.
- fix(#827,#828,#830): identifiability audit / canonical / compiler correctness.
- fix(#863): sphere GPU terms compile (`gpu_err!` import / macro scope).

Cleanup — removed deprecated aliases / compatibility shims (use the canonical names):
- Removed CLI family spelling aliases, shared precision-key aliases, Gumbel schedule aliases, and the identifiability-warning compatibility mirror; unified and restricted GPU policy parsing; reject stale SAE payload shapes.

Tests:
- test(#860, #820, #819): regression pins for startup-validation never-fail escalation, the fuzzer scenario cost cap, and the sparse-exact REML `group()`-panel repro.

## gam v0.3.104 / gamfit v0.1.178

- fix(#787/#785): C1 antiderivative for floored Jeffreys eigenvalue + line-search early-exit threshold (bernoulli marginal-slope inner KKT now converges; centers=12 PGS config that previously froze ~20min now returns).
- fix(#859): pin CTN cross-fit response knot count across folds (skewed-PGS large-scale calibration no longer raises p1 mismatch).
- fix(#813/#821): freeze ALO influence_scale/phi per fit (te() outer-REML no longer grinds; value<->gradient consistency).
- wip(#808): eta1-channel cross-block reduction for survival marginal-slope.

## v0.3.103 — gam 0.3.103 / gamfit 0.1.177 (2026-06-07)

- feat(#817): skew-aware Gamma observation (prediction) intervals. Response-scale predictive bands for the Gamma family are now equal-tailed quantiles of a moment-matched Gamma predictive — built on a robust inverse regularized incomplete-gamma (`probability::gamma_quantile`) — instead of the symmetric `μ ± z·σ` band that systematically mis-covered each tail of a right-skewed response. Includes a per-tail coverage regression and a degenerate-shape symmetric fallback.
- feat(#811, #812): the binomial posterior-mean `predict` path now honours `covariance_mode` (the smoothing correction reaches the credible band) and `observation_interval=True` (emits `observation_lower` / `observation_upper`), matching the Gaussian path and centred on the bias-corrected posterior-mean point; `covariance_mode='required'` now hard-errors when no correction is available.
- fix(#815, #816): `cyclic()` / `cc()` / `cp()` honour `period=` / `origin=` (parsed through the numeric-expression grammar, with a hard error on unparseable endpoints) and validate their options instead of silently falling back to the observed data range.
- fix(#685–#688): Gaussian location-scale fits through the formula API (`noise_formula=`) now converge on heteroscedastic data instead of aborting outer REML on every seed — the log-σ (scale) block carries a REML-selected identity ridge constraining its polynomial nullspace, and the spurious full-span Jeffreys term is dropped. (The hand-built custom-family location-scale path, #684, remains a tracked log-σ recovery / convergence gap.)
- fix(survival): Royston–Parmar monotonicity is enforced at every observed exit time; structural model construction is split from the ≥1-event fittability check so all fit modes share one validation chokepoint.
- fix(multinomial): the matrix-free Hessian diagonal mirrors the dense path's parallel reduction order, restoring bit-identical agreement under IEEE-754 non-associativity.
- fix(reml): the TK-refinement scale gate is aligned with the outer Firth gate.
- fix(#795): `sae_manifold_fit` defaults `isometry_weight=0.0`, so the single-planted-circle quickstart converges (the MeanProfiled isometry energy is not scale-invariant and was saturating the arrow-Schur proximal ridge); pass `isometry_weight > 0` to opt back in. Adds a default-exercising regression test.
- chore: removed accidentally-committed repro scripts / build log from the tree (and the gamfit sdist).

## gam v0.3.102 / gamfit v0.1.176

- fix(#789B): fast-fail survival marginal-slope on all-censored (zero-event) designs instead of spinning.
- fix(#808): never-fail outer escalation for survival marginal-slope (graceful degradation instead of fatal IntegrationFailed); permanent regression test (#814).
- Includes #700-703 tensor/sz null-space penalty defaults, #811/#812 binomial predict covariance_mode/observation_interval threading, #795 periodic-axis shrinkage fix, and #735/#736 log_sigma block-width fixes.

# Changelog

All notable, user-visible changes to **gam** (the Rust engine, published to
crates.io) and **gamfit** (the Python wheel, published to PyPI) are recorded
here. This file is the single source of truth for release notes:

- It is rendered on the documentation site under **Changelog**
  (`docs/changelog.md` includes this file verbatim via a snippet).
- The GitHub Release for each `v*` tag is generated from the matching section
  below.
- CI (`.github/workflows/publish.yml`) refuses to publish a tag whose version
  does not match the top entry here *and* the versions in `Cargo.toml` /
  `pyproject.toml`.

The two packages are versioned independently — `gam` tracks the Rust engine,
`gamfit` the Python wheel — but released together. Each entry is headed with the
git tag and both package versions.
Failed or unpublished version-bump tags are intentionally omitted; package
releases without local semver tags are included under their published version.

## v0.3.101 — gam 0.3.101 / gamfit 0.1.175 (2026-06-06)

- gamfit: survival marginal-slope baseline-hazard conditioning + monotonicity-domain tolerance fixes (#788, #797 inner barrier) reach PyPI; bernoulli marginal-slope inner trust-region noise-floor fix. Plus mainline fixes #798–#805.

## v0.3.100 — gam 0.3.100 / gamfit 0.1.174 (2026-06-06)

Correctness fixes to response-scale prediction intervals, survival fitting, and
the REML/LAML evidence path.

### Fixed
- **Observation (prediction) intervals are clamped to the response support
  (#800).** `predict(..., observation_interval=True)` builds the response-scale
  predictive band as the symmetric `μ ± z·σ_pred`. For a bounded or
  half-bounded response — a count (Poisson, Negative-Binomial, Tweedie), a
  positive value (Gamma), or a proportion (Beta, Binomial) — that band crossed
  the support edge at a small or extreme fitted mean and reported impossible
  values (a Poisson predictive lower bound going negative). The band is now
  floored/capped at the family's response support in both interval-assembly
  paths. The *mean* (confidence) interval was already correct and is unchanged.
  A new `ResponseFamily::response_support_bounds` exposes the closed support
  bounds, kept in lockstep with the existing support-membership check.
- **Beta prediction intervals use the estimated precision φ̂ (#801).** The
  observation-interval builder's Beta arm read precision off the family-enum
  construction seed (default `1.0`) instead of the precision estimated jointly
  with the mean, so on high-precision data the band was `√((1+φ̂)/2)` too wide.
  It now routes through the same fitted-dispersion accessor the Tweedie/Gamma
  arms already use, falling back to the seed only for raw-covariance sources
  that carry no fitted scale.
- **Survival fits no longer over-reject censored rows.** The per-row
  monotonicity guard in the survival working-model update rejected any row whose
  stabilized exit derivative fell below a numerical floor, but only event rows
  evaluate `ln(deriv)` / `1/deriv` downstream. The guard is now gated on event
  rows (`d > 0`), matching the residual-channel loop, so a censored row with a
  zero collocation derivative at, e.g., β = 0 is no longer a false-positive
  monotonicity violation.
- **REML/LAML evidence integrity on indefinite per-row blocks.** In evidence
  (log-determinant) mode the arrow–Schur per-row factorization silently lifted
  the ridge on a non-positive-definite `H_tt` until it became PD, then summed
  the lifted factor's diagonal into the exact arrow log-determinant — reporting
  `log|H_tt + ridge_eff·I|` where `log|H_tt + ridge_t·I|` was intended and
  corrupting the evidence with no error surfaced. Evidence mode now returns a
  typed error on a genuinely non-PD block instead of accepting a ridge-lifted
  surrogate; the strict Newton-step path, which wants the regularising lift, is
  unchanged.

## v0.3.99 — gam 0.3.99 / gamfit 0.1.173 (2026-06-06)

Completes the link-general Firth work from 0.3.98.

### Fixed
- **Non-logit Firth fits no longer crash (#758).** 0.3.98 opened the CLI/HMC
  Firth gate to every Binomial inverse link with a Fisher-weight jet (Probit,
  CLogLog, Latent-CLogLog, SAS, Beta-Logistic, Mixture), but the REML outer
  loop's Tierney-Kadane correction is implemented only for the canonical
  Binomial Logit jet — so `gam fit --firth --family binomial-probit` (or
  cloglog) aborted every outer seed with "Tierney-Kadane outer Hessian is
  implemented for canonical Binomial Logit Firth fits only". Non-logit Firth
  fits now skip the higher-order TK refinement and fall back to plain Laplace
  REML driven by BFGS off the link-general gradient; the Firth/Jeffreys bias
  reduction itself (the inner PIRLS Jeffreys penalty) is fully retained. Logit
  Firth fits are byte-unchanged and keep the full analytic TK path.

## v0.3.98 — gam 0.3.98 / gamfit 0.1.172 (2026-06-06)

First published release carrying the universal under-identification robustness
work staged in 0.3.97 (which was version-bumped but never published to
crates.io / PyPI), together with a batch of correctness fixes across the
binomial-link, separation-diagnostic, SAE-penalty, and constraint paths.

### Fixed
- **Link-general Firth / Jeffreys (#758).** Firth bias reduction and the
  Jeffreys prior now apply to every Binomial inverse link that carries a
  Fisher-weight jet (Probit, CLogLog, Latent-CLogLog, SAS, Beta-Logistic, and
  anchored Mixture links), not only Logit. The actual inverse link is preserved
  through the Firth/Jeffreys and PIRLS-diagnostic paths instead of silently
  collapsing to a bare logit, and the CLI gate and the NUTS/HMC Firth guards
  accept the full set with an accurate message (was a stale "only supported for
  Binomial Logit").
- **Pre-fit regularity screen (#775).** Designs that are perfectly separated
  (single-column *or* linear-combination separators) or rank-deficient in their
  unpenalized block are now rejected up front with a typed, actionable error
  instead of diverging inside the solver.
- **Multinomial separation diagnostics (#753).** Separating multinomial fits
  raise a dedicated `MultinomialSeparationDetected` error naming the offending
  class/row instead of reusing the binary-outcome message.
- **Tweedie sampling dispersion (#771).** Posterior sampling draws Tweedie
  responses with the fitted dispersion φ rather than mis-using the variance
  power as the noise scale.
- **Picklable Rust exceptions (#773).** `gamfit`'s Rust-originated exceptions
  carry an importable module, so they survive pickling across a
  `ProcessPoolExecutor` / `multiprocessing` boundary.
- **SAE-penalty curvature correctness (#794).** `MonotonicityPenalty::hvp` no
  longer inflates curvature by `1/smoothing_eps`, and `JumpReLUPenalty`'s PSD
  majorizer now genuinely dominates the exact (indefinite) Hessian for inactive
  coordinates instead of under-estimating it ~7×.
- **SAE numerical robustness (#742).** Learnable penalty/SAE exponents are
  clamped to a finite-normal band so extreme ρ can no longer overflow to
  inf/NaN.
- **Box-constraint scaling (#791).** CLI box constraints are transformed by
  `1/scale` (not `scale`), fixing constraint escape under non-unit scaling.
- **Random-effect group axes (#792).** Unseen string random-effect group levels
  are admitted at predict time and group axes are no longer clipped.
- **Coupled Dirichlet joint Hessian (#729)** and spec-aware joint-Hessian drifts
  that keep batched marginal-slope / Jeffreys fits Hφ-consistent (#787).
- **Periodic tensor B-splines (#629).** Tensor B-spline periodicity is preserved
  through freeze→reload, with a separable periodic top-1 fit path and a restored
  manifold-SAE serialization roundtrip.
- **Guidance fixes:** correct per-diagnosis marginal-slope refusal guidance
  (#754) and survival marginal-slope penalty-width provenance (#788).

### Internal
- Unified the four matrix-free Lanczos paths onto one primitive (#766),
  collapsed the duplicated Firth-support predicates onto a single source of
  truth, synced the `gam-pyffi` lockfile version, and made the release tree
  rustfmt-clean.

## v0.3.97 — gam 0.3.97 / gamfit 0.1.171 (2026-06-05)

Universal under-identification robustness — unified, always-on, no flag.

### Added
- Robustness is now an unconditional solver property, self-limiting so it is
  byte-identical on well-identified fits and only acts where the data is
  near-separating / under-identified: a conditioning-gated full-span
  Jeffreys/Firth prior on the identifiable subspace (finite estimates under
  separation), a self-gating penalized-complexity prior on the smoothing
  parameters, and exact orthogonalization of confounded design blocks. A cheap
  matrix-free (Lanczos) conditioning pre-check keeps it ~zero-cost and
  matrix-free-preserving on well-conditioned and large-`p` fits.
- Never-fail inference: when the smoothing optimizer cannot certify convergence,
  the fit escalates to sampling the proper posterior (HMC) — guarded by R-hat /
  ESS so it returns honest (never false-confident) uncertainty instead of erroring.

### Changed
- The `RobustIdentification` flag and the pinned BMS nullspace / overlap ridges
  are removed; robustness is a single always-on path with no user knob.

### Fixed
- Published the current `main` engine through the Python wheel line, including
  the random-effect prediction schema fix for unseen string groups and the
  manifold-SAE serialization roundtrip fix.

## v0.3.96 — gam 0.3.96 / gamfit 0.1.169 (2026-06-05)

First crates.io release of the `gam` engine since v0.3.91, bringing the Rust
crate current with the gamfit 0.1.164–0.1.168 wheel line and adding the new
under-identification robustness layer (off by default).

### Added

- **Universal under-identification robustness (`robust_identification`, preview — off by default).** A new, family-general layer that makes robustness to non-identification a property of the *solver* rather than a per-family patch: a link-general Jeffreys/Firth penalty on the under-identified subspace (bounding near-separating coefficients) plus exact orthogonal reparameterisation of overlapping design blocks (resolving structural confounds rather than penalising them). Exposed as `gamfit.fit(..., robust_identification=...)` and the `--robust-identification` CLI flag with policies `"off"` (default), `"auto"`, and `"force"`. **`"off"` is byte-identical to the previous solver**, so existing fits are unchanged; the machinery is opt-in while it is hardened.

### Fixed

- **Smooth-free (purely parametric) fits no longer crash.** An ordinary linear model — `gamfit.fit(df, "y ~ x1 + x2")`, any family, with no `s()`/`te()`/`matern()` term — aborted in the post-fit null-space metadata step with `null-space Hessian is not positive definite: Cholesky factorization failed: NonPositivePivot { index: 0 }`, even though the fit converged and the CLI fit the same data fine. Root cause: a smooth-free design has an all-zero penalty matrix, and the rank-revealing QR returned a NaN null-space basis for rank-0 input (faer's column-pivoted QR produces degenerate Householder reflectors when the first pivot column has zero norm). The null space of a zero matrix is the whole space, so its basis is now returned as the exact identity. This also unblocked learned-length-scale Matérn BMS fits, whose outer optimiser was being poisoned by the same NaN at degenerate penalty configurations.
- **`bs="sz"` factor smooths fit and predict (#700).** A sum-to-zero factor smooth `s(g, x, bs="sz")` crashed at fit time with an identifiability-transform dimension mismatch and was non-functional; the full-design joint-null rotation is no longer folded into the per-marginal `sz` metadata, so `sz` smooths now fit and reproduce their fitted values on frozen replay.
- **Hybrid Duchon smooths with an explicit `length_scale` build for every covariate dimension (#750).** `duchon(...)` with a `length_scale` but no explicit `power=` crashed at basis generation for even covariate dimensions `d ≥ 4`; the cubic structural default now resolves to an admissible integer spectral power.
- **BMS spatial-`rho` startup and convergence hardening (#754, #461).** Fixed `#754`/`#461` ridges are carried as physical `PenaltyMatrix::Fixed` penalties (excluded from the REML/outer `rho` vector), the startup no longer mis-classifies a phantom seed, and production-shaped Matérn BMS fits start and converge.

## gamfit 0.1.168 — gam 0.3.95 / gamfit 0.1.168 (2026-06-05)

### Fixed

- **Publish the post-0.1.167 BMS spatial/kappa rho fix to PyPI (#754).** The
  0.1.167 wheel was built before the follow-up fix that removed fixed physical
  BMS ridges from the learned spatial/kappa REML `rho` layout. This wheel bump
  publishes the already-merged code needed by the large-scale Workbench driver, whose
  `uv --with gamfit --upgrade-package gamfit` path resolves from PyPI.

## v0.3.95 — gam 0.3.95 / gamfit 0.1.167 (2026-06-04)

### Fixed

- **Probit BMS marginal-slope: release the marginal/logslope overlap ridge (#754, completing the fix).** The #754 nullspace-shrinkage ridge (shipped in 0.1.165/0.1.166) bounds the marginal block's *unpenalized* directions, but a production-scale run (`duchon(PC1,PC2,PC3,centers=20)`, n≈195k, 1:1 balanced) showed the runaway coefficient (β≈61) actually lives on a **penalized smooth** direction that is degenerate with the score-weighted logslope surface — the marginal↔logslope confound — which the nullspace ridge does not touch. This release ships the additional fixed **overlap ridge** that shrinks exactly those cross-channel directions, plus the production-shaped binary-outcome BMS regression test. The nullspace ridge alone was necessary but not sufficient at scale; the two ridges together bound both the null-space and the confound directions.

## v0.3.94 — gam 0.3.94 / gamfit 0.1.166 (2026-06-04)

### Fixed

- **`matern(..., centers=K)` no longer FATALs when K over-specifies the kernel (#755).** With a fixed `length_scale`, packing more centers into the data cloud than the kernel can resolve makes adjacent basis functions near-identical, so the realized design carries exactly linearly-dependent columns and the identifiability audit hard-FATALs on intra-block rank deficiency. The basis now rank-reduces the center set at construction: column-pivoted RRQR on the realized `n×K` kernel design (the same matrix the audit checks) at the crate-standard tolerance, keeping the leading full-rank pivoted centers and dropping the redundant remainder (logged). Detection is on the realized design columns (not the squared center Gram), so it fires exactly when the audit would have failed and leaves well-specified bases untouched.

## v0.3.93 — gam 0.3.93 / gamfit 0.1.165 (2026-06-04)

### Fixed

- **Probit Bernoulli marginal-slope outer REML no longer diverges (#754).** The marginal-surface block left its parametric + smooth-nullspace directions fully unpenalized, so on a balanced steep-gradient probit sample a near-separating direction's coefficient ran to ~50 and the outer ARC solve hit max-iter / rejected every seed (`phantom_multiplier_with_well_conditioned_H`) — basis-independent (Matérn and Duchon both hit it). A small **fixed** nullspace-shrinkage ridge (`Z·Zᵀ` over the null space of the aggregate marginal smooth penalties), pinned out of REML at `log λ = ln(1e-2)` so it cannot be driven to zero, now bounds the flat direction and gives the outer solve a finite optimum — negligible against the n-scaled probit Fisher information of any identified direction.

## v0.3.92 — gam 0.3.92 / gamfit 0.1.164 (2026-06-04)

### Changed

- Release bump to force a fresh wheel build/publish. No engine changes since 0.1.163; in-progress fixes for the Bernoulli marginal-slope outer-REML non-convergence (#754) and Matérn over-parameterization (#755) will follow in a later release.

## v0.3.91 — gam 0.3.91 / gamfit 0.1.163 (2026-06-04)

### Fixed

- **Binary-outcome-style Bernoulli marginal-slope Matérn fits now have full audit-level regression coverage.** The release includes a formula-to-fit test for the reported `matern(...) + sex + entry_age_z + current_age_ns_*` layout, proving the scalar-pruned model passes the actual pre-fit identifiability audit and produces finite coefficients.

## v0.3.90 — gam 0.3.90 / gamfit 0.1.162 (2026-06-04)

### Fixed

- **Binary-outcome-style Bernoulli marginal-slope formulas now have an exact regression for scalar-alias pruning.** The release includes a materialization test matching the reported `matern(...) + sex + entry_age_z + current_age_ns_*` layout and proves the local-column-3 scalar alias is removed before the identifiability audit while the Matérn blocks remain intact.

## v0.3.89 — gam 0.3.89 / gamfit 0.1.161 (2026-06-04)

### Fixed

- **Bernoulli marginal-slope redundant-scalar handling now has fail-closed
  regression coverage.** Tests now lock in that constrained or explicitly
  penalized duplicate scalar columns are rejected rather than pruned, preserving
  the hardened identifiability audit contract for binary-outcome-style BMS
  formulas with redundant scalar covariates.

## v0.3.88 — gam 0.3.88 / gamfit 0.1.160 (2026-06-04)

### Fixed

- **Release metadata for the Bernoulli marginal-slope identifiability fix is now complete.** The PyPI wheel crate and lockfile now carry the same `gamfit` version as `pyproject.toml`, satisfying the hardened release scanner for the BMS redundant-scalar audit fix shipped in the previous commit.

## v0.3.87 — gam 0.3.87 / gamfit 0.1.159 (2026-06-04)

### Fixed

- **Bernoulli marginal-slope Matérn fits with redundant scalar covariates no longer fail the identifiability audit.** The workflow now removes unpenalized scalar columns that add no direction beyond the implicit intercept and earlier scalar terms before BMS block construction, and rejects constrained or explicitly-penalized duplicates instead of using a ridge or constraint to mask non-identifiability. This keeps the hardened audit fail-closed while allowing large-scale binary-outcome-style `matern(...) + scalar covariates` fits whose precomputed scalar spline column is constant or redundant.

## v0.3.86 — gam 0.3.86 / gamfit 0.1.158 (2026-06-04)

### Fixed

- **The #751 survival marginal-slope release is now build-gate clean and wired
  through PyPI metadata.** The release lockfile now carries the current
  `gamfit` version, custom-family output-channel defaults use the real
  single-output channel map instead of a sentinel empty-spec shortcut, and SAE
  fixed-decoder projection grids are selected from the atom basis kind rather
  than mandatory evaluator methods with no-op `None` implementations.

## v0.3.85 — gam 0.3.85 / gamfit 0.1.157 (2026-06-04)

### Fixed

- **Survival marginal-slope left-truncated `matern(...)` fits no longer reject
  every REML seed through a phantom time-block multiplier** (#751). The
  marginal-slope baseline time basis now anchors at the median exit time instead
  of the minimum entry time, so left truncation no longer turns the centered
  I-spline null-space column into a dominant one-sided time trend. The time block
  also installs an explicit null-space shrinkage penalty for structural
  unpenalized directions, giving REML a real precision parameter instead of an
  unidentifiable phantom multiplier.
- **Invalid survival marginal-slope custom-family block specs now return a typed
  error instead of panicking in Rust** (#751). Output-channel wiring validates the
  block specs before probing family channel assignments, and the default
  assignment hook is only defined for empty specs.

## v0.3.84 — gam 0.3.84 / gamfit 0.1.156 (2026-06-04)

This release lands a large batch of correctness, convergence, and quality fixes
across families, plus a build fix that restores the PyPI wheel and crates.io
publish paths.

### New

- **Held-out split-conformal calibration fold** (#682). The conformal prediction
  path now accepts a calibration fold whose size differs from the training set.
  The fold is routed through plain split-conformal — residuals on the held-out
  fold, normalized by the predict-time response-scale SE — instead of being
  bound to the training set's frozen ALO geometry, so calibrating on a fold of
  any size produces finite, coverage-valid intervals.
- **GPU CUDA userspace preload.** On Linux, the CUDA userspace libraries
  (cudart, nvJitLink, cuBLAS, cuSPARSE, cuSOLVER) are preloaded with
  `RTLD_GLOBAL` from canonical toolkit directories and pip `nvidia-*-cu12` wheel
  layouts, so cudarc's lazy SONAME loads resolve without an `LD_LIBRARY_PATH`
  mutation. Discovery is environment-variable-free (interpreter-relative, plus
  the wheel's `$ORIGIN`-relative rpath).

### Fixed

- **Gaussian location-scale predictions are reported in raw response units.**
  The model standardizes the response internally (keeping the log-σ soft floor
  scale-relative) and now maps the fitted coefficients, covariance, and
  likelihood/deviance/REML summaries back to raw units before persistence;
  prediction no longer applies a second response-scale multiplier, so the mean
  and σ come out in the data's own units with the response scale applied
  exactly once.
- **Survival location-scale constant-scale AFT fits no longer hang, panic, or
  fail finalization** (#735, #736, #721). A constant-scale parametric AFT now
  builds an identifiable parametric time-warp that absorbs the unidentified
  I-spline null space; the constrained joint-Newton QP is damped on a
  rank-deficient `H_pen` so an unidentified time-warp gauge step exhausts and
  the identified-subspace KKT certificate fires (instead of crawling a dead-flat
  REML ridge); the parametric-AFT time `ρ` is seeded at the inner box bound so
  the box-constraint KKT certifies immediately; the log-σ canonicalization keeps
  an intercept-only scale block at width 1 so raw/active block widths agree at
  the covariance-lift boundary; and the batched-outer `state.eta` length check
  uses `solver_design().nrows()` rather than `design.nrows()` (3·n vs n),
  fixing a finalization panic.
- **Continuous-transformation-normal/monotone (CTN/CTM) fits converge and report
  EDF** (#720, #733, #734). The response-basis size adapts to the
  transformation's non-normality and the inner exact-Newton cycle cap is scoped
  to the bounded convex block (no more dense exact-SCOP-Hessian timeout on a
  simple Gaussian shift); a rank-deficient penalized-Hessian null direction no
  longer blocks KKT certification (the identified range-space certificate is a
  first-class per-cycle test, and the CTM joint-Newton range-projects the RHS
  instead of erroring); a self-vanishing Levenberg damping stabilizes the inner
  spectral Newton step; and total EDF / inference are populated for joint-Newton
  custom-family fits.
- **Coupled multi-block custom families are trusted without an explicit marker**
  (#727, #729). A family that returns a genuinely coupled (nonzero
  off-diagonal-block) joint Hessian is detected structurally and used, rather
  than requiring `has_explicit_joint_hessian()`.
- **Tensor `te()`/`ti()` and factor smooths no longer over-smooth** (#700, #701,
  #702, #703, #712, #713). The tensor double penalty shrinks only the joint null
  space rather than applying a full identity ridge; `sz` factor smooths drop the
  inner-marginal double penalty so each per-level linear null space stays free
  like mgcv; and `fs` factor smooths cap the marginal basis to the least-resolved
  group so per-group curves shrink toward a linear random slope. The `te()`
  capacity guard also sums marginal column counts instead of their Kronecker
  product, so well-posed penalized tensors on moderate `n` are accepted
  (#724, #728, #730).
- **ALO stabilization no longer over-smooths under high leverage** (#711), and
  logistic link-scale confidence intervals are calibrated by pooling across
  Bernoulli replicates to estimate Nychka across-the-function coverage (#710).
- **Competing-risks CIF reconstruction uses the fitted baseline** rather than the
  seed configuration (#689, #690).
- **Tweedie variance-power estimation stays inside the valid `(1, 2)` interval**
  (#698). The biased 6-bin log-variance OLS slope is replaced with a
  saddlepoint profile-likelihood MLE bounded to the open interval, so the
  estimated power no longer escapes to ~2.29 on real data.
- **Binomial probit recovery** (#697) is benchmarked against a method-comparable
  penalized GLMGam reference (via the predict path, no invalid `.scale` access),
  closing the RMSE gap to statsmodels' probit fit.
- **Bernoulli marginal-slope reports an honest terminal verdict** (#744). A fit
  that stalls no longer returns after the final cycle with the residual still
  above tolerance without saying so; the joint-Newton terminal criterion is now
  named explicitly.
- **Negative-binomial and Gamma-log standard errors** (#679) gain behavioral
  coverage and cross-family SE guards.
- **Spatial smooths**: by-factor thin-plate-spline length-scale auto-init now
  recurses into the by/factor inner kernels so the predict design is finite
  (#704); 2-D TPS spatial fits use a data-proportional center floor to avoid a
  timeout (#718).
- **Manifold SAE**: scale-free (mean-profiled) isometry reference, an exact
  von-Mises ARD normalizer (Bessel I₀) for periodic axes (#681), EM routing-seed
  refinement for cold multi-atom fits (#629, #630), preserved `random_state`
  seed-dependence through the routing seed (#178), and finiteness/robustness
  guards — log-space IBP prior, clamped learnable weights, NuclearNorm
  active-rank cap, and a Welford PCA seed (#742).
- **Build**: restored the gam-pyffi / maturin compile (a stray unqualified
  `SaeManifoldTerm` reference) and cleared the workspace ban-gate violations in
  the CUDA preload path, so the PyPI wheel and crates.io publish workflows build
  again.

### Notes

- Recalibrated several quality-suite bounds to attainable match-or-beat-reference
  targets: multinomial-logit recovery (#699), p-spline interior-gap (#708),
  sphere/torus SOS surfaces and doubly-cyclic tensors (#694, #695, #705),
  binomial-probit (#697), and the `fs` random-slope lme4 reference DGP (#712).
- Right-sized the nightly real-data posterior-sampling budget (Pólya-Gamma Gibbs
  + PyMC NUTS) and enabled PyMC chain parallelism (#719).

## v0.3.83 — gam 0.3.83 / gamfit 0.1.155 (2026-06-04)

### Fixed

- **Anisotropic Duchon spatial terms no longer abort REML with an outer
  gradient-length mismatch.** Four functions disagreed on how many `ψ` entries a
  multi-axis `aniso_log_scales` Duchon term contributes to the joint outer
  hyperparameter vector: the n-block exact-joint spatial optimizer planned a
  per-axis `θ` layout (`rho_dim + Σ d_term`) while the inner unified evaluator
  emitted one `ψ` per term, tripping the `OuterThetaLayout` contract and failing
  every nightly Large-scale `duchon16d` shard before the solver even started.
  A single shared predicate now drives the `ψ` count at all four sites: Duchon
  anisotropy `η` is a fixed, geometry-derived basis parameter (one isotropic
  `ψ̄` slot per term), so the outer plan and the inner gradient agree by
  construction. Matérn anisotropy is unchanged (still per-axis `ψ`).
- **Manifold SAE fits converge again across all isometry/topology cells,
  including the circle `d=1` case that regressed in 0.1.154.** The isometry
  cross-block curvature added for 0.1.152 left the coordinate/decoder Schur
  complement slightly non-PD — an inconsistent nonzero cross-block that was not
  paired with diagonals from the same residual Jacobian — so circle `d=1` fits
  failed where they had recovered `R² ≈ 0.997`. The inconsistent cross-block is
  removed (PSD diagonals with a zero cross-block stay PSD), and inner
  stationarity is now judged by the gradient at the step's parameter scale so
  gauge-like SAE directions are no longer mistaken for non-convergence. All nine
  isometry × topology × dimension bisection cells now converge.

## v0.3.82 — gam 0.3.82 / gamfit 0.1.154 (2026-06-03)

### Fixed

- **Nuclear-norm HVP no longer panics on roundoff-scale smoothed Gram
  eigenvalues.** The penalty now returns explicit errors for invalid spectra and
  floors only numerical roundoff to the configured smoothing floor, preventing a
  Rust panic from aborting Python SAE experiments.

## v0.3.81 — gam 0.3.81 / gamfit 0.1.153 (2026-06-03)

### Fixed

- **SAE decoder-incoherence convergence checks now fail loudly.** The cross-atom
  decoder cross-Gram test no longer uses `pytest.xfail`; failed or degenerate
  multi-atom fits now surface as ordinary test failures.
- **Python wheel publishing no longer carries stale gamfit references.** Updated
  the lockfile and REML benchmark guidance to the current `gamfit` version so
  the release ban gate accepts the wheel build.

## v0.3.80 — gam 0.3.80 / gamfit 0.1.152 (2026-06-03)

### Fixed

- **Manifold SAE isometry curvature now includes the coupled coordinate/decoder
  Gauss-Newton cross block.** The Arrow-Schur system can now add dense analytic
  `H_tβ` supplements on top of the matrix-free row operator, so the isometry
  metric penalty contributes consistently to `H_tt`, `H_tβ`, and `H_ββ` instead
  of leaving the Schur complement with a missing cross term.
- **Matrix-free and dense SAE cross-block curvature now compose deterministically.**
  `ArrowSchurSystem` fingerprints dense `H_tβ` supplements when they are active,
  and all apply/materialize/transpose paths sum the matrix-free and dense pieces.

### Notes

- This release intentionally keeps the strict SAE KKT gradient tolerance. It does
  not include the earlier experimental tolerance relaxation that made low-quality
  isometry fits appear converged.

## v0.3.79 — gam 0.3.79 / gamfit 0.1.151 (2026-06-03)

### New

- **Cross-atom decoder incoherence for manifold SAE fits**
  (`decoder_incoherence_weight`, #671). A separability lever for multi-atom
  dictionaries: for `K >= 2` it is on by default and penalizes overlap between
  *co-activating* atoms' decoder column spaces, weighted by empirical gate
  co-activation. The penalty now also enters the SAE REML selection criterion,
  so it shapes both the fit and topology/model selection (previously it only
  influenced the Newton step).
- **Decoder embedding-rank selection for manifold SAE fits**
  (`nuclear_norm_weight`, `nuclear_norm_max_rank`, #672). A positive weight
  applies a nuclear-norm penalty to each atom's decoder block, shrinking its
  singular spectrum to select the ambient embedding dimension;
  `nuclear_norm_max_rank` caps the number of leading singular values included.
- **Non-convex SCAD/MCP gate sparsity for manifold SAE fits.** Set
  `gate_sparsity="scad"` or `"mcp"` (with `scad_mcp_gamma` defaulting to `3.7`
  for SCAD and `2.5` for MCP). The default `gate_sparsity="l1"` path is
  unchanged.
- **Per-atom posterior shape uncertainty on `ManifoldSAE` results.** Atoms carry
  `decoder_covariance`, `shape_band_coords`, `shape_band_mean`, and
  `shape_band_sd`; helpers `shape_uncertainty(...)` and the `shape_band(...)`
  alias expose the posterior shape band.
- **Typical coordinate-range summaries for manifold SAE atoms.**
  `coordinate_range(...)` gives per-axis min/max/median/5th/95th-percentile
  summaries; `typical_shape(...)` restricts the posterior shape band to an
  atom's typical recovered-coordinate range.

### Fixed

- **Intrinsic, gauge-invariant decoder smoothness for SAE topology evidence**
  (#673). The decoder roughness penalty is now reparameterized into arc length
  via the decoder pullback metric `g = JᵀJ` (a symmetric congruence of the raw
  penalty), so the `reml_score` used to compare an atom's topology (e.g. circle
  vs. line) is invariant to reparameterizing the latent coordinate.
  Constant-speed and periodic atoms are provably unchanged. Previously the
  penalty was computed in raw latent coordinates, making topology evidence
  gauge-dependent for non-constant-speed atoms.
- **Gamma dispersion is no longer over-estimated (~2×) when the mean varies**
  (#678). The Gamma shape `ν = 1/φ` was frozen at an early, far-from-converged
  linear predictor. It is now re-estimated at the converged `η` and iterated to
  the joint `(β, ν)` fixed point — only at the single final reported fit at the
  REML-selected `λ`, so the smoothing-parameter search is unaffected.
- **Standard errors for Gamma, Tweedie, Beta, and Negative-Binomial models are
  no longer too small by √dispersion** (#679). The coefficient covariance
  `Vb = H⁻¹` is no longer multiplied by a post-hoc dispersion factor for
  families whose IRLS working weight already carries the dispersion / full
  Fisher information; only the profiled Gaussian restores `Vb = H⁻¹·σ̂²`. Encoded
  as a single-source-of-truth invariant
  (`GlmLikelihoodSpec::coefficient_covariance_scale`).
- **More accurate SAE reconstruction dispersion `φ̂`** (#676). The
  latent-coordinate effective degrees of freedom now use the exact ARD-shrunk
  trace instead of the full assignment-weighted latent dimension, so posterior
  shape bands are no longer mildly conservative.
- **Manifold SAE multi-atom routing no longer collapses to a uniform saddle**
  (#629, #630). Cold-start assignment logits are seeded asymmetrically from the
  per-atom reconstruction residual (an EM-style step) instead of exactly
  uniform, which was a symmetric saddle for `K >= 2` exchangeable atoms. The
  outer REML search also now rejects finite-but-non-converged inner solves
  rather than ranking them.
- **Out-of-sample SAE encoding recovers one-hot periodic-atom routing** (#628).
  A global decoder-projection coordinate seed places each row in the correct
  basin before refinement, and the OOS path keeps the decoder frozen. The torus
  projection-seed grid now falls back to a PCA seed past its point cap instead
  of emitting an exponentially large grid.
- **SAE inner-solver convergence regressions** that could surface as
  `RemlConvergenceError`. The arrow-Schur PCG `schur_matvec` callback clears its
  reused output buffer before accumulating `S·x`, preventing stale contributions
  from corrupting the reduced system.
- **SAE joint arrow-Schur line-search baseline.** The solver snapshots the exact
  state used to assemble the gradient and Hessian and computes `pre_step_total`
  from it before Armijo backtracking, so trial steps are no longer compared
  against a stale objective.
- **Non-Linux builds** now provide a real `scatter_batched`, so targets that
  call it unconditionally compile; device-free runs report no device tiles and
  the caller runs its deterministic whole-batch CPU fallback.

### Verified

- Verified the per-atom shape-uncertainty plumbing end-to-end (Python ↔ PyO3 ↔
  Rust) and the analytic Schur block-inverse identity used for the posterior
  bands (#677).

## v0.3.78 — gam 0.3.78 / gamfit 0.1.151 (2026-06-03)

### Changed

- Published the changelog/docs/release-wiring pass together with the #679
  coefficient-covariance-scale fixes and SAE intrinsic-roughness work that
  preceded the tagged `v0.3.79` repair.
- Wired the root `CHANGELOG.md` into docs, PyPI project URLs, and GitHub Release
  note generation; the next release corrected stale `gamfit` version references.

## v0.3.77 — gam 0.3.77 / gamfit 0.1.150 (2026-06-03)

### Changed

- Added global decoder-projection coordinate seeding for fixed-decoder SAE
  out-of-sample prediction (#628).

## v0.3.76 — gam 0.3.76 / gamfit 0.1.149 (2026-06-03)

### Fixed

- Fixed unseen random-effect level prior variance to use `scale / lambda`, not
  `1 / lambda`, and capped CI linker parallelism so concurrent release links do
  not exhaust runner memory (#674).

## v0.3.75 — gam 0.3.75 / gamfit 0.1.148 (2026-06-03)

### Changed

- Added ManifoldSAE per-atom posterior shape uncertainty end-to-end: Rust
  decoder covariance and shape bands, PyO3 exposure, Python result fields, and
  an e2e regression test.
- Expanded GPU/Arrow-Schur execution with per-ordinal `AtB` GEMM, multi-GPU
  row-block solves, shared manifold kernels, Schur inverse-block extraction, and
  all-GPU manifold batch GEMM/GEMV dispatch.
- Fixed non-Linux GPU `scatter_batched` / ordinal `AtB` paths, response-scale
  invariance for smooth Wald p-values (#675), leftover merge-conflict markers,
  and pyffi build drift.

## v0.1.147 — gam 0.3.74 / gamfit 0.1.147 (2026-06-02)

### Changed

- Re-triggered the gamfit wheel after the 0.3.74 / 0.1.146 coordinated release;
  the next `v0.3.75` release carried the remaining code changes.

## v0.3.74 — gam 0.3.74 / gamfit 0.1.146 (2026-06-02)

### Changed

- Moved dense-Fisher multi-output Gaussian fitting, block-orthogonal REML
  backward, Fisher-Rao weight normalization, SPD/symmetric solves, weighted
  ridge solving, auxiliary-prior REML scoring, and SAE PCA seeding from the FFI
  layer into core Rust.
- Exposed conformal intervals, covariance-mode / observation prediction, and
  Wood per-smooth p-values through `gamfit`.
- Fixed release compilation across targets, benchmark Rust-extension loading,
  BMS/SMGS exact-joint probe-design reconstruction, Circle wrapping docs, and a
  broad geometry/SAE audit batch (#596-#626).

## v0.1.145 — gam 0.3.73 / gamfit 0.1.145 (2026-06-02)

### Changed

- Published the intermediate gamfit wheel between `v0.3.72` and `v0.3.74`; the
  substantive core/FFI changes are captured in the adjacent coordinated
  releases.

## v0.3.72 — gam 0.3.72 / gamfit 0.1.144 (2026-06-02)

### Fixed

- Fixed PSIS Zhang-Stephens GPD shape estimation so heavy-tail `k_hat` is not
  capped at 0.5 (#585).
- Fixed response-scale-equivariant `Vp` / effective-n dispersion, logit erfcx
  quadrature derivatives, periodic 1-D Duchon PSD Bernoulli kernels, isotropic
  Matern divergence gates, `sz` continuous-first row sizing, SAE warm-start
  reuse / PSD Arrow-Schur ridge conditioning, top-k SAE encoder backprop, and
  auxiliary-conditional identifiability rank scaling (#576-#584).
- Folded the skipped `v0.3.70` and `v0.3.71` work into this published release.

## v0.1.143 — gam 0.3.71 / gamfit 0.1.143 (2026-06-01)

### Changed

- Published the second intermediate wheel in the `v0.3.70` / `v0.3.71` series;
  its fixes are folded into the `v0.3.72` notes.

## v0.1.142 — gam 0.3.70 / gamfit 0.1.142 (2026-06-01)

### Changed

- Published the first intermediate wheel in the response-scale / SAE /
  identifiability repair series that was consolidated in `v0.3.72`.

## v0.3.69 — gam 0.3.69 / gamfit 0.1.141 (2026-06-01)

### Changed

- Landed the draft GPU survival-FLEX row-primary gradient/Hessian launcher and
  oracle, and reduced survival marginal-slope per-row influence-absorber
  allocation churn.
- Reworked reference-quality CI with per-test wall-clock budgets, INLA
  dependency provisioning, a clearer outcome taxonomy, and right-sized
  expensive mgcv/LOO/scipy/tram tests.
- Fixed `ti(2d)` main-effect leak measurement, compositional-mean quality gates,
  badhealth `te()` mgcv references, row-Hessian non-Linux gating, and deleted
  the dead `dynamic_q_core_hessian_blocks` path.
- Published the actual `gamfit` 0.1.141 wheel version.

## v0.3.68 — gam 0.3.68 / gamfit 0.1.140 (2026-05-31)

### Fixed

- Fixed active-set scale invariance, bounded-shape Jacobian reporting, sphere
  latitude clipping at prediction time, and monotone-shape REML startup
  regressions (#500, #507, #508, #509).

## v0.3.67 — gam 0.3.67 / gamfit 0.1.139 (2026-05-31)

### Fixed

- Fixed published-crate build hygiene, exact Circle exp-map behavior,
  cubic-cell derivative consistency, and survival marginal-slope row-context
  error reporting.

## v0.3.65 — gam 0.3.65 / gamfit 0.1.136 (2026-05-29)

### Changed

- Restored main to a buildable state after lint/import drift and added the
  streaming SAE joint-fit path: minibatch on-demand recompute, block-sparse atom
  Schur structure, row-procedural GPU `H_tβ` matvecs, on-device Jacobi-CG, and a
  real scaling/parity demo (#358).
- Fixed REML penalty-coordinate projection onto the active-set free subspace,
  Gamma scaled-deviance likelihood use in the outer objective, multinomial REML
  deviance reuse, AdaptiveTopK hard top-k behavior, and ManifoldSAE config-matrix
  joint-solve coverage (#347-#360).
- Removed dead Gamma likelihood and sparse penalty imports left by the K>=2 SAE
  mechanism-sparsity refactor.

## v0.1.137 — gam 0.3.65 / gamfit 0.1.137 (2026-05-29)

### Fixed

- Fixed the identifiability anchor-correction dimension invariant and made it a
  release check before the later attempted `v0.3.66` / `gamfit 0.1.138` release.

## v0.3.64 — gam 0.3.64 / gamfit 0.1.135 (2026-05-29)

### Fixed

- Restored a coherent 0.3.x Rust release line after Python-only tags.
- Fixed ManifoldSAE out-of-sample inference to reuse fit-time SAE
  hyperparameters, replaced the static SAE basis shim with real Duchon/Euclidean
  basis refresh, and repaired positional isometry pairing with loud Jacobi
  non-convergence.
- Fixed deep-tail inverse-link precision for cloglog/probit-related paths,
  restored Linux GPU imports/macros, routed latent multi-output GLMs through
  canonical fitters, and added penalized multi-binomial family entry points.
- Removed three dominant CPU costs from the SAE inner Newton loop and repaired
  test/CI agent workflow drift.

## v0.1.134 — gam 0.2.3 / gamfit 0.1.134 (2026-05-28)

### Changed

- Re-triggered the gamfit wheel after the 0.1.131-0.1.133 version-bump attempts;
  the next coordinated `v0.3.64` release carried the corrected Rust and Python
  package state.

## v0.2.3 — gam 0.2.3 / gamfit 0.1.128 (2026-05-25)

### Fixed

- Published the CUDA runtime / diagnostics release between the 0.2.x and 0.3.x
  lines: exposed Python CUDA diagnostics, added loader tests, fixed cudarc
  CPU-only-host behavior, tightened runtime diagnostics, and shipped substantial
  Bernoulli marginal-slope / custom-family / REML-eval fixes.

## v0.1.130 — gam 0.2.3 / gamfit 0.1.130 (2026-05-27)

### Changed

- Published the gamfit 0.1.130 wheel after the 0.1.129 test-refactor release,
  before the later failed 0.1.131-0.1.133 bump attempts.

## v0.1.129 — gam 0.2.3 / gamfit 0.1.129 (2026-05-27)

### Fixed

- Published test refactors and minor fixes on the 0.2.3 engine line, including
  follow-up cleanup after the CUDA runtime diagnostics release.

## v0.3.63 — gam 0.2.2 / gamfit 0.1.124 (2026-05-25)

### Changed

- Coordinated the Rust/Python release after the 0.1.122 wheel line,
  including tensor B-spline derivative scratch reuse, JumpReLU logit-init
  fixes, Rust prediction-helper exports, and a large documentation accuracy
  pass across getting-started, predictions, REML scaling, persistence, sklearn,
  and GPU acceleration docs.
- Expanded regression coverage around SAE manifold flow, periodic basis
  validation, production linearized residuals, and doc examples.

## v0.3.62 — gam 0.2.1 / gamfit 0.1.123 (2026-05-25)

### Fixed

- Fixed periodic `basis_with_jet` shape validation, macOS `dynamic_lookup`
  pyffi linking, production-match tolerances for the red regression checks, and
  SAE manifold flow over the newer Rust auto APIs / prediction helpers.

## v0.1.122 — gam 0.2.1 / gamfit 0.1.122 (2026-05-24)

### Fixed

- Fixed SAE-manifold periodic Fourier basis behavior, added hard out-of-sample /
  multi-seed / sphere accuracy regressions, and replaced skipped OOS checks with
  assertions that the prediction surface exists.
- Rebalanced survival marginal-slope stall repro cohorts, tightened panic
  assertions, disabled GPU on macOS for the stall repro, and removed unused
  Python-extension imports.

## v0.1.121 — gam 0.3.65 / gamfit 0.1.121 (2026-05-24)

### Changed

- Added CV/permutation evidence documentation and bumped gamfit to 0.1.121; the
  next worked wheel line is captured by `v0.1.122`.

## v0.1.120 — gam 0.3.65 / gamfit 0.1.120 (2026-05-23)

### Changed

- Completed the Tweedie / Negative-Binomial exhaustive support pass, added
  Tweedie log-link inference handling, stabilized SAE Gumbel temperature/log
  weights, tightened sparse-exact solve handling, removed stale SAE fallbacks,
  and deleted stale composition-engine / audit-log material.

## v0.1.119 — gam 0.3.65 / gamfit 0.1.119 (2026-05-23)

### Changed

- Added negative-binomial likelihood variants, PIRLS support, log-link inference
  across HMC/sampling, and latent Negative-Binomial support.
- Refactored Arrow-Schur core solve paths, broadened latent-basis dispatch,
  added per-point Hessian artifacts, reused latent REML jets for topology
  evidence gradients, and corrected SAE IBP sign / ext-coordinate naming.

## v0.1.118 — gam 0.3.65 / gamfit 0.1.118 (2026-05-23)

### Changed

- Shipped Tweedie likelihood support, Euclidean metric-weighted reduced-Schur
  trust-region solves, Tweedie latent-GLM plumbing, exact latent-cache
  invalidation on latent updates, and all-target build unblocks for
  `StandardFitRequest`, dead-code, and the Ceres scaffold.

## v0.1.117 — gam 0.3.65 / gamfit 0.1.117 (2026-05-23)

### Fixed

- Fixed the `latent_cache.rs` REML import path and refreshed proposal docs for
  the landed latent-coordinate and iVAE pieces.

## v0.1.116 — gam 0.3.65 / gamfit 0.1.116 (2026-05-23)

### Changed

- Owned tensor knot slices in pyffi, wired Rust SAE IBP fitting, added latent
  design caching, latent ID direct hyperparameters, derivative/jet-backed basis
  evaluation, safe missing-cache handling for isometry penalties / ARD log
  terms, and stricter latent/SAE validation with fallible penalty builders.

## v0.1.115 — gam 0.3.65 / gamfit 0.1.115 (2026-05-23)

### Fixed

- Fixed the 0.1.114 wheel-build failure, added per-axis manifold metric weights
  and `ProductWithMetric`, normalized `fisher_w` naming, and added the Ceres
  backend scaffold.
- Carried the composition-engine WIP from the failed 0.1.114 tag forward:
  latent-coordinate plumbing, Arrow-Schur analytic penalties, Riemannian metric
  weighting, IBP-MAP / SAE-manifold pieces, topology-selection helpers, and
  strict mkdocs link/anchor fixes.

## v0.1.113 — gam 0.3.65 / gamfit 0.1.113 (2026-05-23)

### Fixed

- Routed the unified outer Hessian projected-operator path through the
  K-pseudoinverse, tied matrix-free stochastic-trace flags to materialization
  budgets, fixed a CTN `effective_weights` recursion bug, and aligned Duchon
  hybrid auto-resolution with `max_op=2`.
- Accepted mgcv-style relative-to-cost convergence for spatial iso/aniso fits
  that stop on `max_iter`.

## v0.1.112 — gam 0.3.65 / gamfit 0.1.112 (2026-05-22)

### Changed

- Exposed hybrid Duchon spectral knobs (`length_scale`, `nullspace_order`, and
  `power`) through Python primitives, added high-dimensional/default-argument
  regression coverage, fixed `difference_smooth(group_means=False)` to target
  the group main effect, expanded formula/smooth/family docs, made outer
  gradient norms optional in fit results, and removed stale sentinel docs.

## v0.1.111 — gam 0.3.65 / gamfit 0.1.111 (2026-05-22)

### Fixed

- Re-tagged the gamfit wheel with a type-annotation fix for HGB helpers and
  handled ill-conditioned REML backward passes with a zero-gradient fallback;
  documented and benchmarked `gt.fit` mode dispatch scaling.

## v0.1.110 — gam 0.3.65 / gamfit 0.1.110 (2026-05-22)

### Changed

- Added automatic dispatch between joint and independent torch fitting,
  constrained Gaussian REML backward through torch autograd, projected
  Firth/Jeffreys logdet paths, stability-gated sensitivity allocation, and
  stricter runtime budgeting.
- Reset outer IFT residual caches per fit and guarded trust-energy gates against
  stale cached residuals.

## v0.1.109 — gam 0.3.65 / gamfit 0.1.109 (2026-05-22)

### Changed

- Added analytic multi-block REML backward / VJP support, exposed block APIs,
  routed Gaussian REML forward through simpler wiring, and fixed multi-block
  REML gradients.
- Applied row-mask weighting in survival Hessian paths and replaced weighted
  cross-products with masked `mxtwx` psi multiplications.

## v0.1.108 — gam 0.3.65 / gamfit 0.1.108 (2026-05-22)

### Changed

- Added per-smooth lambda additive REML support and Smooth API exposure,
  including torch additive REML routing to the per-smooth multi-block solver,
  term diagnostics / block REML outputs, HT outer subsampling support for
  Gaussian and binomial location-scale paths, IFT warm-start beta prediction,
  and periodic / sphere basis APIs.

## v0.1.107 — gam 0.3.65 / gamfit 0.1.107 (2026-05-22)

### Fixed

- Handled `Result<usize, SurvivalError>` from survival event-code cause counts,
  surfaced typed errors through pyffi, added PIRLS AA(1) Fisher acceleration,
  exposed IFT residual metrics in `RemlLamlResult`, and suppressed
  envelope-inconsistent gradients unconditionally.

## v0.1.106 — gam 0.3.65 / gamfit 0.1.106 (2026-05-22)

### Changed

- Exposed `duchon_function_norm_penalty` as a public helper and continued the
  adaptive PIRLS KKT / outer-gradient integration.
- Returned typed errors rather than panics for invalid survival event codes,
  enforced contiguous survival event codes, and tightened REML derivative
  contracts.

## v0.1.105 — gam 0.3.65 / gamfit 0.1.105 (2026-05-22)

### Changed

- Published multi-dimensional Duchon and additive REML APIs, replacing 1-D
  Duchon bindings with `duchon_basis`, removing legacy Duchon derivative
  exports, and adding additive REML output / wrappers.
- Migrated torch tests to the new multi-D Duchon and additive REML API, fixed
  pyffi compile drift from core refactors, used typed link/distribution payload
  fields, added projected-KKT certificate regressions, and tightened constrained
  stationarity certification.

## v0.1.104 — gam 0.3.65 / gamfit 0.1.104 (2026-05-21)

### Fixed

- Used a rank-thresholded pseudo-inverse for the active-constraint Schur
  complement and bumped gamfit from 0.1.103 to 0.1.104.

## v0.1.103 — gam 0.3.64 / gamfit 0.1.103 (2026-05-21)

### Changed

- Switched ALO/HMC, scale-design, smooth/REML numerics, term builders, and
  custom-family active-constraint assembly to typed errors.
- Plumbed active inequality constraints through unified REML inner assembly,
  blockwise inner results, and blockwise active-constraint propagation; removed
  unused active-constraint RHS storage and centralized family floors.

## v0.1.102 — gam 0.3.63 / gamfit 0.1.102 (2026-05-21)

### Changed

- Removed unused saved-link helpers and dead link fallbacks, stored survival
  distributions as typed enums, preferred explicit saved survival links, and
  normalized monotone-root errors into a typed error.
- Routed REML coordinate solves through the penalty-subspace kernel, added
  checked diagonal block working sets / `SymmetricMatrix` helpers, documented
  GPU acceleration and CUDA stack conflicts, and locked the warn-not-raise
  contract for CUDA dual-stack detection.

## v0.1.101 — gam 0.3.62 / gamfit 0.1.101 (2026-05-21)

### Changed

- Refactored log-link IRLS, likelihood-family checks, link-state validation,
  SPD Levenberg-Marquardt logdet continuation, posterior quadrature helpers,
  and strict eta/clamp constants into shared code.
- Required projected KKT residuals for joint-Newton REML paths, refined PIRLS
  convergence certificates, documented `by` semantics for Gaussian REML
  position fits, and removed synthbug conflict remnants.

## v0.1.100 — gam 0.3.61 / gamfit 0.1.100 (2026-05-21)

### Changed

- Published the gamfit 0.1.100 wheel after the trust-region diagnostics line;
  the following `v0.1.101` release carried the shared log-link / likelihood /
  REML refactor.

## v0.3.61 — gam 0.3.61 / gamfit 0.1.99 (2026-05-21)

### Changed

- Classified and surfaced trust-region radius decisions in diagnostics, added
  joint-Newton stall labels, linearized residual metrics, logdet Hessian test
  derivatives, and outer-scale soft convergence exits for PIRLS.

## v0.3.60 — gam 0.3.60 / gamfit 0.1.98 (2026-05-21)

### Fixed

- Raised the custom-family default `inner_max_cycles` from 100 to 300, rejected
  boundary-saturated cache seeds, added screening proxy evaluation, and
  normalized survival marginal-slope inner-fit options.

## v0.3.59 — gam 0.3.59 / gamfit 0.1.97 (2026-05-21)

### Fixed

- Discarded fully saturated cached `rho` values instead of clamping them, added
  fractional Duchon null-space tests, tightened covariance-shape
  classification, clamped nonpositive survival times, and stabilized
  competing-risks CIF endpoint assembly / FFI returns.

## v0.3.58 — gam 0.3.58 / gamfit 0.1.96 (2026-05-21)

### Changed

- Shipped the IFT projected pseudo-inverse fix for gamfit and refactored
  competing-risks prediction payloads.
- Added competing-risks CIF / prediction / paired-sampling APIs, shared
  precision cross-fit helpers with fitted lambdas, fractional polyharmonic
  Duchon order support, hard covariance tests, and broader survival hard-test
  coverage.

## v0.1.95 — gam 0.3.57 / gamfit 0.1.95 (2026-05-21)

### Fixed

- Passed Duchon operator block order as `f64` at all call sites and carried the
  envelope-gradient / boundary-rho cache fixes into the next worked gamfit
  wheel line.

## v0.1.94 — gam 0.3.57 / gamfit 0.1.94 (2026-05-20)

### Fixed

- Short-circuited outer-Hessian assembly when the envelope-gradient check would
  trip, clamped cached boundary `rho` seeds, and removed unused KKT residual
  plumbing.

## v0.3.57 — gam 0.3.57 / gamfit 0.1.92 (2026-05-20)

### Changed

- Published REML optimization work including geometric Hessian scaling, CUDA
  diagnostics, cached inner warm-start state, exact-hit cache
  short-circuiting, deduplicated projected GEMMs, fused REML accumulation slice
  fast paths, faer-backed eigenbasis rotations, and parallel chunk traversal
  outside rayon pools.

## v0.3.56 — gam 0.3.56 / gamfit 0.1.91 (2026-05-20)

### Fixed

- Threaded EFS Hessian scale through eval samples and barrier checks, gated EFS
  on relative barrier curvature, fixed survival marginal-slope accumulator /
  scale-jet / joint-psi second-order wiring, skipped redundant warm-start
  pilots by family fingerprint, and pinned `opt` to the registry release.

## v0.1.90 — gam 0.3.56 / gamfit 0.1.90 (2026-05-20)

### Changed

- Published the package bump immediately before the EFS Hessian-scale release;
  the substantive EFS barrier/eval-sample wiring shipped in `v0.3.56`.

## v0.3.55 — gam 0.3.55 / gamfit 0.1.89 (2026-05-20)

### Fixed

- Published the CUDA diagnostics / KKT-convergence bundle between `v0.3.54` and
  `v0.3.56`: explicit Python CUDA diagnostic wrappers, CUDA stack conflict
  tests, row-kernel work modeling for rigid survival outer Hessians, stricter
  PIRLS KKT residual scaling, non-convergence / line-search failure handling,
  and canonical block-local Gaussian penalty-logdet derivatives.

## v0.1.88 — gam 0.3.55 / gamfit 0.1.88 (2026-05-20)

### Changed

- Added CUDA stack diagnostics and conflict checks, plus Python CUDA diagnostic
  wrappers, before the `v0.3.55` package-alignment release.

## v0.3.54 — gam 0.3.54 / gamfit 0.1.87 (2026-05-20)

### Fixed

- Added cuBLAS/CUDA dual-load diagnostics and defenses, including preload
  ordering, complete CUDA-stack validation, persisted `libcublas` handles, and
  removal of the process-level Python GPU disable hook.
- Refreshed GPU/survival/posterior documentation, torch extras metadata, and
  marginal-slope visualization assets.

## v0.1.86 — gam 0.3.54 / gamfit 0.1.86 (2026-05-20)

### Changed

- Refreshed package metadata and lockfile state for the CUDA dual-load defense
  line before the `v0.3.54` coordinated release.

## v0.3.53 — gam 0.3.53 / gamfit 0.1.85 (2026-05-19)

### Changed

- Re-tagged the Bernoulli periodic Duchon work with gamfit 0.1.85 so the wheel
  publish path ran.

## v0.1.85 — gam 0.3.52 / gamfit 0.1.85 (2026-05-19)

### Changed

- Bumped gamfit from 0.1.84 to 0.1.85 for the Bernoulli periodic Duchon wheel
  release.

## v0.3.52 — gam 0.3.52 / gamfit 0.1.84 (2026-05-19)

### Fixed

- Fixed Bernoulli periodic Duchon kernels and expanded position-basis alias
  coverage.
- Dropped duplicate periodic endpoint centers, enforced odd effective K in seam
  tests, used the Bernoulli Green's kernel, covered design rank and `B_4`
  spectrum, removed Duchon PSD projection helpers, and required strict KKT
  residuals for joint inner convergence.

## v0.3.51 — gam 0.3.51 / gamfit 0.1.83 (2026-05-19)

### Fixed

- Raised survival pilot caps, improved dense trust-region steps, fixed REML
  penalty/log-lambda gradients, used objective-scaled absolute gradient floors
  for outer convergence certification, and simplified solution-certification
  logging.

## v0.3.50 — gam 0.3.50 / gamfit 0.1.82 (2026-05-19)

### Changed

- Shipped the convergence-truthfulness bundle: objective-floor guards,
  post-convergence status reporting, and no silent success on stalled fits.
- Added Python REML scoring APIs, `grad_penalty` output for
  `gaussian_reml_score`, non-REML smoothing support in batched position fits,
  free-coefficient position REML scoring, basis alias normalization, Duchon
  function-norm penalties, and tighter REML penalty routing.

## v0.3.49 — gam 0.3.49 / gamfit 0.1.81 (2026-05-19)

### Changed

- Added the batched psi-term fast path for survival marginal-slope fits, with a
  regression test that batched terms match per-axis terms, plus batch correction
  and ext-coordinate Hessian solves in solver/REML.

## v0.3.48 — gam 0.3.48 / gamfit 0.1.80 (2026-05-19)

### Fixed

- Enabled automatic outer subsampling by default for marginal-slope fits,
  failed survival marginal-slope fits on outer non-convergence, removed invalid
  survival-prediction row fallbacks, handled droppable NaN rows explicitly, and
  required a KKT residual ceiling for flat-step PIRLS convergence.

## v0.3.47 — gam 0.3.47 / gamfit 0.1.79 (2026-05-19)

### Fixed

- Broadened CUDA preload paths, simplified CUDA `dlopen` warning text, used
  mode ridge for predicted reduction in blockwise trust-region, tightened flat
  joint-step convergence, and added rho=2 stabilization proof / saturated-null
  direction diagnostics.

## v0.3.46 — gam 0.3.46 / gamfit 0.1.78 (2026-05-19)

### Changed

- Added always-on visualizer sessions for `gamfit` fits, open-ended workflow
  progress feeds, optimizer metrics and cost sparklines, fixed-log-lambda
  survival pilot warm starts, CUDA calibration diagnostics, and cudarc-backed
  CUDA / cuBLAS preflight checks.
- Tightened inner-solve convergence handling, removed the Bernoulli step cap,
  and surfaced GPU calibration errors.

## v0.3.45 — gam 0.3.45 / gamfit 0.1.76 (2026-05-19)

### Changed

- Completed the GPU module migration to cudarc 0.19 and exposed Python-side GPU
  activity / visualizer state.
- Reduced PIRLS joint-Newton log verbosity and moved accepted-cycle timing to
  debug output.

## v0.1.75 — gam 0.3.44 / gamfit 0.1.75 (2026-05-19)

### Changed

- Added hierarchical near-match warm starts and made cache keys survive package
  version bumps.
- Preflighted `libcuda` / `libcublas` loads before cudarc calls, migrated cuBLAS
  runtime and CUDA transfers to cudarc wrappers, added cache mirror-session
  finalization broadcast, threaded survival cache sessions, and accepted
  saturated hazard / convergence regimes.
- Aligned Bernoulli cross-block orthogonalization with the PIRLS Hessian metric
  and limited PIRLS near-convergence log promotion to residual convergence.

## v0.1.74 — gam 0.3.44 / gamfit 0.1.74 (2026-05-19)

### Fixed

- Added load-side finiteness gates for caches, throttled joint-Newton logs, and
  hardened GPU/cache paths, including missing ndarray imports in the GPU session
  path and parallel GPU-session safety.

## v0.1.73 — gam 0.3.44 / gamfit 0.1.73 (2026-05-19)

### Changed

- Made warm starts uniform across custom-family fits and wired cache-session
  hooks through fit requests and solver entry points.
- Kept GPU design matrices resident across PIRLS iterations, warmed the GPU
  runtime early, exposed Python GPU activity summaries, calibrated dispatch
  thresholds from measured runtime metrics, added blockwise cache-session
  options, and ran survival regression / save-load roundtrip tests in CI.

## v0.1.72 — gam 0.3.44 / gamfit 0.1.72 (2026-05-19)

### Fixed

- Fixed survival time-basis persistence so saved models always include the
  anchor and the construction path imports `SavedSurvivalTimeBasis`.

## v0.1.71 — gam 0.3.44 / gamfit 0.1.71 (2026-05-19)

### Changed

- Persisted survival time-basis snapshots, populated marginal-slope payload
  baseline/time-basis fields, and covered the saved `survival_time` basis field.

## v0.1.70 — gam 0.3.44 / gamfit 0.1.70 (2026-05-19)

### Changed

- Added position REML basis-state outputs, auto-resolved position basis inputs,
  sped col-major conversion, and routed memory-limited marginal-slope chunks to
  CPU when the GPU path is ineligible.
- Exposed Rust 1-D automatic basis placement for `None`/integer knots and
  centers, returned basis state through payload attachments, lowered GPU GEMM /
  GEMV / TRSM dispatch thresholds, and adjusted joint-objective acceptance for
  floating-point roundoff.

## v0.1.69 — gam 0.3.44 / gamfit 0.1.69 (2026-05-19)

### Fixed

- Released the 0.1.69 wheel after the 0.3.44 Rust engine line and removed the
  remaining `xfail` markers from the full 34/34 torch suite.

## v0.1.68 — gam 0.3.44 / gamfit 0.1.68 (2026-05-19)

### Changed

- Canonicalized Gaussian REML penalties, required symmetric torch REML
  penalties, added cache mismatch diagnostics, and expanded torch/REML
  regression coverage.
- Symmetrized penalty gradients in closed-form REML backward, stabilized EDF
  backward gradcheck matrices, treated tiny analytic/finite-difference gradients
  as zero in relative-error helpers, logged each GPU routing signature once, and
  removed debug markers from Gaussian REML ill-conditioning paths.

## v0.1.67 — gam 0.3.44 / gamfit 0.1.67 (2026-05-18)

### Changed

- Published the final pre-`v0.1.68` gamfit-only wheel after the 0.3.44 engine
  bump, carrying the torch REML and package-layout stabilization work that the
  next tagged release made explicit.

## v0.1.66 — gam 0.3.44 / gamfit 0.1.66 (2026-05-18)

### Changed

- Published a gamfit-only wheel on the 0.3.44 engine line after the 0.3.44 /
  pyffi 0.1.66 / gamfit 0.1.64 coordinated bump.

## v0.1.64 — gam 0.3.44 / gamfit 0.1.64 (2026-05-18)

### Changed

- Bumped the engine and Python bridge to the 0.3.44 / 0.1.66 line, preparing
  the torch REML symmetric-penalty release series that culminated in
  `v0.1.68`.

## v0.1.62 — gam 0.3.42 / gamfit 0.1.62 (2026-05-15)

### Changed

- Published the follow-up package bump after the 0.3.41 / 0.1.61 release,
  keeping Rust, pyffi, and gamfit versions aligned during the May 15 torch/REML
  release train.

## v0.1.61 — gam 0.3.41 / gamfit 0.1.61 (2026-05-15)

### Changed

- Published the next coordinated Rust / pyffi / gamfit bump after the 0.1.60
  wheel, preserving package alignment for the torch/REML work.

## v0.1.60 — gam 0.3.40 / gamfit 0.1.60 (2026-05-14)

### Changed

- Published the first worked package bump after the hard-pseudo REML Hessian /
  joint-solver rejection series, moving the engine toward the later 0.3.44 torch
  REML release line.

## v0.3.36 — gam 0.3.36 / gamfit 0.1.56 (2026-05-11)

### Fixed

- Fixed PyPI release workflow skip propagation by prefixing the release job
  condition with `always()`.
- Folded in the skipped `v0.3.35` workflow-dispatch publish fix so manual PyPI
  release runs actually reach the release job.

## v0.3.34 — gam 0.3.34 / gamfit 0.1.54 (2026-05-11)

### Fixed

- Fixed the PyPI `workflow_dispatch` release-job gate.

## v0.3.33 — gam 0.3.33 / gamfit 0.1.53 (2026-05-11)

### Fixed

- Translated `expects N blocks, got 0` block-state shape errors at the Python
  boundary instead of surfacing raw Rust messages.
- Carried forward the skipped `v0.3.32` belt-and-suspenders guard against empty
  survival location-scale `block_states`.

## v0.3.31 — gam 0.3.31 / gamfit 0.1.51 (2026-05-11)

### Fixed

- Projected the REML outer-gradient trace kernel for non-Gaussian families,
  addressing the root `expects 3 blocks, got 0` failure.

## v0.3.30 — gam 0.3.30 / gamfit 0.1.50 (2026-05-11)

### Fixed

- Tightened the custom-family outer `rho` bound to prevent ARC-stall crashes.

## v0.3.29 — gam 0.3.29 / gamfit 0.1.49 (2026-05-11)

### Fixed

- Broadened the survival location-scale empty `block_states` crash guard.

## v0.3.28 — gam 0.3.28 / gamfit 0.1.48 (2026-05-11)

### Changed

- Tuned the PyPI `release-pypi` profile max-runtime settings.
- Folded in the skipped `v0.3.26` / `v0.3.27` release-gate changes, including
  the on-demand `linux_only` PyPI release input and package metadata bumps.

## v0.3.25 — gam 0.3.25 / gamfit 0.1.45 (2026-05-11)

### Changed

- Version-only publish marker after the survival location-scale optimizer
  update.

## v0.3.24 — gam 0.3.24 / gamfit 0.1.44 (2026-05-11)

### Changed

- Replaced survival location-scale baseline `CompassSearch` with an
  analytic-gradient BFGS path.

## v0.3.23 — gam 0.3.23 / gamfit 0.1.43 (2026-05-11)

### Changed

- Sped up survival location-scale GM baseline profiling, preserved benchmark
  shard output across blocking failures, and marked the flexible Rust GAM
  benchmark as non-blocking.
- Added a spatial-kappa REML re-evaluation drift tolerance and removed, then
  restored, mathematically infeasible joint-PC Duchon benchmark scenarios as the
  benchmark blocking policy changed.

## v0.3.22 — gam 0.3.22 / gamfit 0.1.42 (2026-05-11)

### Changed

- Added native ARM Linux / cache-warming PyPI workflow support, refreshed
  marginal-slope documentation figures, and preserved in-flight family and
  prediction-path fixes.
- Deleted the orphaned `approx_ledger` module and its test, dropped tracked
  benchmark result artifacts, and tightened benchmark-result gitignore rules.

## v0.3.21 — gam 0.3.21 / gamfit 0.1.41 (2026-05-11)

### Changed

- Routed location-scale and latent survival sampling to a Laplace fallback and
  added the first MkDocs/Material documentation site.
- Rewrote the README, added Read the Docs / Material configuration, docs CI,
  social assets, and broader `gamfit` docstrings for API reference rendering.

## v0.3.20 — gam 0.3.20 / gamfit 0.1.40 (2026-05-11)

### Changed

- Threaded `PseudoLogdetMode` through the matrix-free SPD operator.

## v0.3.19 — gam 0.3.19 / gamfit 0.1.39 (2026-05-11)

### Fixed

- Fixed PyPI workflow invocation by dropping the mutually exclusive `--release`
  flag from the profile build.

## v0.3.18 — gam 0.3.18 / gamfit 0.1.38 (2026-05-11)

### Changed

- Added a BLAS-3 `projected_matrix` override with output symmetrization and an
  `n * rank^2` threshold gate, and sped up PyPI wheel CI.

## v0.3.17 — gam 0.3.17 / gamfit 0.1.37 (2026-05-11)

### Fixed

- Fixed BMS batched `dH` rayon-pool starvation deadlocks at small row counts.

## v0.1.37 — gam 0.3.17 / gamfit 0.1.37 (2026-05-11)

### Fixed

- Fixed the same BMS batched `dH` starvation issue for the Python-wheel tag and
  carried marginal-slope / GAMLSS performance gates forward.

## v0.3.15 — gam 0.3.15 / gamfit 0.1.34 (2026-05-11)

### Fixed

- Fixed the gam-pyffi saved-runtime payload by adding missing
  `SavedAnchoredDeviationRuntime.anchor_residual` fields.
- Folded in the skipped `v0.3.14` Duchon derivative work: design and frozen-Z
  penalty finite-difference tests, a no-identifiability variant, and a 1-D
  `power=2` linear-control probe.

## v0.3.13 — gam 0.3.13 / gamfit 0.1.32 (2026-05-11)

### Changed

- Added `RayonSafeOnce` for lazy caches whose initialization dispatches nested
  rayon work.

## v0.3.12 — gam 0.3.12 / gamfit 0.1.31 (2026-05-11)

### Fixed

- Fixed ARC retry config borrowing and removed temporary diagnostic output.

## v0.3.11 — gam 0.3.11 / gamfit 0.1.30 (2026-05-11)

### Changed

- Hardened PSD penalty handling, ARC retry behavior, cache fingerprinting,
  per-axis composite traces, and pseudo-inverse reuse for REML Hessians.
- Added workspace-cached performance paths, richer constraint-nullspace errors,
  finite-difference hardening, exact-hit PIRLS LRU clearing on outer-seed
  resets, `opt` 0.5.3 `NumericallyConverged` handling, and certified-final-value
  gates for joint-spatial REML surfaces.

## v0.3.10 — gam 0.3.10 / gamfit 0.1.29 (2026-05-11)

### Changed

- Published the rank-deficient rho-Hessian / BMS diagnostic work between
  `v0.3.9` and `v0.3.11`: consistent observed-Hessian jets, penalty-redundancy
  diagnostics, marginal-slope protocol cleanup that stopped baking in
  `score_warp` / `link_deviation`, and repeated spectral-operator log
  coalescing.

## v0.3.9 — gam 0.3.9 / gamfit 0.1.29 (2026-05-10)

### Changed

- Improved large-scale performance with rank-INT latent-z handling,
  line-search subsampling, row-set threading, and predict-time anchor
  correction.
- Added BMS residual plumbing, cross-block identifiability regressions for
  `(I-P_A)C` residualization, rigid-path performance gates, and adaptive
  GAMLSS inner-cycle caps / soft warm starts across `rho`.

## v0.3.8 — gam 0.3.8 / gamfit 0.1.29 (2026-05-10)

### Changed

- Added Hutch++ trace estimators, two-phase automatic row subsampling, and
  row-kernel pooling for marginal-slope families.
- Added batched `MultiDirJet` contraction for survival marginal-slope
  third/fourth derivatives, KKT-on-null post-step certification, REML
  penalty-rank cliff fixes, entropy-driven fuzz seeding, and hot-path speedups
  in non-affine cell evaluation / row Hessian construction.

## v0.3.7 — gam 0.3.7 / gamfit 0.1.28 (2026-05-10)

### Changed

- Extended cross-block identifiability to parametric anchors, certified cycle-0
  KKT convergence, and extracted diagonal Hessian scores for joint Newton.
- Refactored the Hutch++ marginal-slope API and removed a probit
  Hessian-collapse test that asserted logit-only behavior.

## v0.3.6 — gam 0.3.6 / gamfit 0.1.27 (2026-05-10)

### Changed

- Wired cross-block identifiability APIs through Bernoulli marginal-slope and
  BMS joint-orthogonal flexible bases.
- Carried coordinated Bernoulli and survival-family edits from the concurrent
  agent work into the release.

## v0.3.5 — gam 0.3.5 / gamfit 0.1.26 (2026-05-10)

### Changed

- Rejected non-converged CTN screening seeds and opted exact-joint initial-rho
  screening into relevant custom-family paths.
- Removed stale CTN screening imports and screened BFGS exact-joint CTN seeds
  before ranking them.

## v0.3.4 — gam 0.3.4 / gamfit 0.1.25 (2026-05-09)

### Fixed

- Stopped capped BFGS on objective stall and shipped the custom-family
  line-search workspace fixes from the concurrent edit range.

## v0.3.3 — gam 0.3.3 / gamfit 0.1.24 (2026-05-09)

### Fixed

- Fixed blockwise PIRLS trust-region adaptation.

## v0.3.2 — gam 0.3.2 / gamfit 0.1.23 (2026-05-09)

### Fixed

- Fixed Windows wheel builds by normalizing paths in the approximate-ledger
  scanner.
- Anchored the CTN BFGS step cap and cleaned up BMS formatting drift.

## v0.3.1 — gam 0.3.1 / gamfit 0.1.22 (2026-05-09)

### Fixed

- Fixed the unused-assignment warnings and accidental `probability.rs`
  truncation that broke 0.3.0 wheel builds.
- Removed redundant joint-Newton rejection bookkeeping and restored rustfmt
  compliance in subsampling / row-construction paths.

## v0.3.0 — gam 0.3.0 / gamfit 0.1.21 (2026-05-09)

### Changed

- Shipped the audit-driven correctness pass: rcond-floor scale-design
  truncation without Tikhonov bias, stable tail likelihoods, honest
  indefinite-Hessian skips, outer-derivative guards on inner convergence,
  Horvitz-Thompson row weights for outer-score subsampling, resource-policy
  auto-derivation, exported Laplace-curvature labeling, and the stabilization
  ledger / spectral classifier cleanup.

## v0.1.20 — gam 0.2.1 / gamfit 0.1.20 (2026-05-09)

### Fixed

- Fixed survival location-scale identifiability, clarified custom-family
  convergence comments, optimized GAMLSS projected traces, and simplified score
  warp anchoring.
- Moved the benchmark suite to nightly-only execution and carried
  Bernoulli/custom-family/GAMLSS/transformation-normal local rollups forward.

## v0.1.19 — gam 0.2.1 / gamfit 0.1.19 (2026-05-09)

### Fixed

- Restored Bernoulli marginal-slope and survival prediction files after a bad
  local rollup, reverted a custom-family joint-line-search workspace hook, and
  tightened inner KKT / finite-difference checks.
- Replaced data-distribution moment-anchor tests with full-rank penalty tests
  and recovered rustfmt compliance in Bernoulli marginal-slope and survival
  prediction code.

## v0.1.18 — gam 0.2.1 / gamfit 0.1.18 (2026-05-09)

### Fixed

- Auto-chunked dense survival predictions and replaced data-distribution
  moment-anchor tests with full-rank penalty tests before the `v0.1.19`
  corrective release.

## v0.2.1 — gam 0.2.1 / gamfit 0.1.17 (2026-05-08)

### Changed

- Added the outer-operator `apply_into` trait hook, trimmed the published crate
  include-list under crates.io limits, and split PyPI wheel build from publish.
- Dropped a GAMLSS cache path that no longer matched the operator API.

## v0.2.0 — gam 0.2.0 / gamfit 0.1.17 (2026-05-08)

### Changed

- Migrated the optimizer integration to `opt` 0.5.0, including
  `DeclaredHessianForm` capability plumbing and accepted-step observation.
- Added `FirthAugmentedSingleHyperOperator::trace_projected_factor`, chunked
  GEMM traces for implicit hyperoperators, and follow-up migration across
  `OuterProblem`, GAMLSS, and mixture-zero estimate paths.

## v0.1.17 — gam 0.1.17 / gamfit 0.1.17 (2026-05-07)

### Changed

- Introduced the top-level `gamfit` Python package layout, removed the
  one-off PGS/examples wrappers, reorganized benches, added PyPI wheel
  publishing, and shipped the cache-policy auto-derivation, marginal-slope
  performance, inner-Newton line-search, and Python-binding cleanup from the
  long pre-release diff.
- Added NUTS posterior sampling exposure, marginal-slope and GAMLSS performance
  work, BMS LRU derivative paths, BLAS-3 Bernoulli rigid-row kernels, large-scale
  design dense-conversion fixes, and broader Python packaging cleanup.

## v0.1.16 — gam 0.1.16 / gamfit 0.1.16 (2026-05-07)

### Changed

- Renamed the Python package to `gamfit`, relicensed to AGPL-3.0-or-later,
  replaced joint-Newton backtracking with a trust-region path, dropped
  pre-whitening from constraint-nullspace solves, and moved the publish workflow
  to token-based publishing.

## v0.1.14 — gam 0.1.14 / gamfit - (2026-03-17)

### Changed

- Completed the sparse Takahashi analytic-gradient line: matrix-level
  perturbation Hessians, exact `P_total` trace gradients, geometry caching,
  Matern parallelism, cross-trace speedups, block-local penalty operations, and
  clean lib/test builds.

## v0.1.13 — gam 0.1.13 / gamfit - (2026-03-16)

### Changed

- Published the large REML / LAML sparse-calculus pass: block-local Takahashi
  traces, sparse exact outer calculus, exact penalty pseudo-logdet on positive
  eigenspaces, structural nullspace dimensions, survival fourth derivatives,
  GAMLSS observed weights, Firth drift fixes, and large-scale PCG /
  compositional hyper-drift work.

## v0.1.11 — gam 0.1.11 / gamfit - (2026-02-25)

### Changed

- Moved crude-risk survival quadrature into the engine, added dual-risk and
  survival regression tests, moved gradient-isolation/parity scripts into the
  Rust test tree, and added core CI / benchmark GitHub Actions workflows.

## v0.1.10 — gam 0.1.10 / gamfit - (2026-02-25)

### Changed

- Version-only crates.io publish marker for the stabilized 0.1.x Rust engine.

## v0.1.9 — gam 0.1.9 / gamfit - (2026-02-25)

### Changed

- Optimized analytic trace contractions and expanded equation-to-code
  derivation comments.

## v0.1.8 — gam 0.1.8 / gamfit - (2026-02-25)

### Changed

- Made the exact survival rho-gradient path the default while retaining a
  finite-difference fallback API.

## v0.1.7 — gam 0.1.7 / gamfit - (2026-02-25)

### Changed

- Added exact rho-gradient survival optimizer path and migrated engine tests to
  it.

## v0.1.6 — gam 0.1.6 / gamfit - (2026-02-25)

### Changed

- Improved external optimizer seed ordering and early-exit behavior.

## v0.1.5 — gam 0.1.5 / gamfit - (2026-02-25)

### Fixed

- Fixed non-Gaussian LAML gradient double counting.

## v0.1.4 — gam 0.1.4 / gamfit - (2026-02-25)

### Fixed

- Fixed external optimizer stationarity handling and evaluated convergence in
  z-space.

## v0.1.3 — gam 0.1.3 / gamfit - (2026-02-25)

### Changed

- Switched external non-Gaussian REML checks to an objective-consistent
  finite-difference gradient.

## v0.1.2 — gam 0.1.2 / gamfit - (2026-02-25)

### Fixed

- Fixed cancellation in non-Gaussian REML gradients at high smoothing
  parameters.

## v0.1.1 — gam 0.1.1 / gamfit - (2026-02-25)

### Changed

- Strengthened oracle tests and aligned survival penalty derivatives.

## v0.1.0 — gam 0.1.0 / gamfit - (2026-02-25)

### Changed

- Initial crates.io release of the Rust GAM engine, including formula/design
  construction, REML/PIRLS fitting paths, survival and non-Gaussian families,
  uncertainty output, and engine test coverage.

## v0.0.0 — gam 0.0.0 / gamfit - (2026-02-24)

### Changed

- Initial placeholder crates.io publication after the early GAM stack import,
  including Duchon/Matern basis coverage, sparse-native REML paths, probit
  location-scale warm starts, model-consistency fixes, and the first Rust CI
  workflow.
