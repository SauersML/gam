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

## gamfit 0.1.168 — gam 0.3.95 / gamfit 0.1.168 (2026-06-05)

### Fixed

- **Publish the post-0.1.167 BMS spatial/kappa rho fix to PyPI (#754).** The
  0.1.167 wheel was built before the follow-up fix that removed fixed physical
  BMS ridges from the learned spatial/kappa REML `rho` layout. This wheel bump
  publishes the already-merged code needed by the AoU Workbench driver, whose
  `uv --with gamfit --upgrade-package gamfit` path resolves from PyPI.

## v0.3.95 — gam 0.3.95 / gamfit 0.1.167 (2026-06-04)

### Fixed

- **Probit BMS marginal-slope: release the marginal/logslope overlap ridge (#754, completing the fix).** The #754 nullspace-shrinkage ridge (shipped in 0.1.165/0.1.166) bounds the marginal block's *unpenalized* directions, but a production-scale run (`duchon(PC1,PC2,PC3,centers=20)`, n≈195k, 1:1 balanced) showed the runaway coefficient (β≈61) actually lives on a **penalized smooth** direction that is degenerate with the score-weighted logslope surface — the marginal↔logslope confound — which the nullspace ridge does not touch. This release ships the additional fixed **overlap ridge** that shrinks exactly those cross-channel directions, plus the production-shaped hypertension BMS regression test. The nullspace ridge alone was necessary but not sufficient at scale; the two ridges together bound both the null-space and the confound directions.

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

- **Hypertension-style Bernoulli marginal-slope Matérn fits now have full audit-level regression coverage.** The release includes a formula-to-fit test for the reported `matern(...) + sex + entry_age_z + current_age_ns_*` layout, proving the scalar-pruned model passes the actual pre-fit identifiability audit and produces finite coefficients.

## v0.3.90 — gam 0.3.90 / gamfit 0.1.162 (2026-06-04)

### Fixed

- **Hypertension-style Bernoulli marginal-slope formulas now have an exact regression for scalar-alias pruning.** The release includes a materialization test matching the reported `matern(...) + sex + entry_age_z + current_age_ns_*` layout and proves the local-column-3 scalar alias is removed before the identifiability audit while the Matérn blocks remain intact.

## v0.3.89 — gam 0.3.89 / gamfit 0.1.161 (2026-06-04)

### Fixed

- **Bernoulli marginal-slope redundant-scalar handling now has fail-closed
  regression coverage.** Tests now lock in that constrained or explicitly
  penalized duplicate scalar columns are rejected rather than pruned, preserving
  the hardened identifiability audit contract for hypertension-style BMS
  formulas with redundant scalar covariates.

## v0.3.88 — gam 0.3.88 / gamfit 0.1.160 (2026-06-04)

### Fixed

- **Release metadata for the Bernoulli marginal-slope identifiability fix is now complete.** The PyPI wheel crate and lockfile now carry the same `gamfit` version as `pyproject.toml`, satisfying the hardened release scanner for the BMS redundant-scalar audit fix shipped in the previous commit.

## v0.3.87 — gam 0.3.87 / gamfit 0.1.159 (2026-06-04)

### Fixed

- **Bernoulli marginal-slope Matérn fits with redundant scalar covariates no longer fail the identifiability audit.** The workflow now removes unpenalized scalar columns that add no direction beyond the implicit intercept and earlier scalar terms before BMS block construction, and rejects constrained or explicitly-penalized duplicates instead of using a ridge or constraint to mask non-identifiability. This keeps the hardened audit fail-closed while allowing biobank hypertension-style `matern(...) + scalar covariates` fits whose precomputed scalar spline column is constant or redundant.

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
  every nightly Biobank-scale `duchon16d` shard before the solver even started.
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

- Improved biobank-scale performance with rank-INT latent-z handling,
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
  work, BMS LRU derivative paths, BLAS-3 Bernoulli rigid-row kernels, biobank
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
  GAMLSS observed weights, Firth drift fixes, and biobank-scale PCG /
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
