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
