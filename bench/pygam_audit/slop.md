# Axis: statistical slop — "do not copy" + "principled replacement"

Auditor scope: pyGAM 0.x (installed in `scratchpad/bvenv`, `site-packages/pygam/`) read as a statistician;
gamfit (repo `/home/user/gam`, installed wheel 0.1.267) checked for the same failure modes.
All demo scripts live in `scratchpad/audit/slop/` (`dNN_*.py`), run with the bvenv python, stdout of gamfit
diagnostics filtered (`grep -v '^\['`). No file under `/home/user/gam` was modified; no cargo build was run.

Note: installed gamfit 0.1.267 differs slightly from repo HEAD (e.g. repo `Model.conditional_aic` ==
installed `Model.evidence` property). Evidence cites repo file:line; demos use the installed wheel.

---------------------------------------------------------------------------------------------------

## Part A — pyGAM methodology to NOT copy (and where gamfit stands)

### P1  Default fit does no smoothing selection at all (lam = 0.6 hard-coded)  — high — already-better
- pyGAM: `terms.py` `SplineTerm(lam=0.6)`, `n_splines=20`; `.fit()` just runs PIRLS at that λ.
  Smoothness is a constant, not an estimate.
- Demo `d02_defaults.py` (A) and `d09_gamfit_null.py` (n=500, σ=0.3):

  | truth       | pyGAM default edof / RMSE | pyGAM gridsearch (GCV) | gamfit REML edf / RMSE |
  |-------------|---------------------------|------------------------|------------------------|
  | pure noise  | 11.98 / 0.0410            | λ=1e3 (grid edge), edof 2.46 / 0.0149 | **0.31 / 0.0042** |
  | linear      | 12.00 / 0.0517            | λ=1e3 (grid edge), edof 2.47 / 0.0355 | **2.38 / 0.0349** |
  | sin(8πx)    | 12.02 / 0.1089            | λ=0.063, edof 15.87 / 0.0518          | 10.91 / 0.1421 (see G1) |

  pyGAM default spends ~12 edf on pure noise; it cannot recover the null (violates SPEC "defaults must
  recover the null"). gamfit fits REML by default and shrinks the null to ~0.3 edf thanks to the double
  (null-space) penalty (`smoothing_parameters` returns 2 λ per smooth).
- gamfit: REML/LAML only, no GCV in production code — only policy comments mention GCV:
  `crates/gam-spec/src/lib.rs:564`, `crates/gam-solve/src/model_types/result_types.rs:1894`,
  `crates/gam-terms/src/structure/anova_atom.rs:324`, `crates/gam-sae/src/k_selection.rs:12`.
- Do not copy: a fixed default λ; any "fit then optionally gridsearch" API split.

### P2  GCV/UBRE with ad-hoc γ = 1.4, optimised on an 11-point log grid  — high — pyGAM-slop-to-avoid
- `pygam.py:1156–1228` (`_estimate_GCV_UBRE`, `gamma=1.4`); `pygam.py:1808–2085` gridsearch,
  default `lam=np.logspace(-3,3,11)`.
- Why wrong: GCV is known to under-smooth with multiple local minima and high variance of λ̂ (Reiss &
  Ogden 2009, Wood 2011); γ=1.4 is a folk fudge that changes the criterion itself; `d01_bugs.py` shows
  the stored GCV = n·D/(n−1.4·edof)² exactly (0.088999802540), i.e. the reported "GCV" is not GCV.
  The 11-point grid has no convergence notion: on noise and linear truth the optimum is the grid
  boundary λ=1e3 (d09), so the answer is an artefact of the grid range. Grid resolution is one decade
  per 0.6 step; EDF is not continuous in the result.
- gamfit: grid-free; e.g. `crates/gam-solve/src/gaussian_reml.rs:4631–4639` replaced a 96-point ρ grid
  by a derivative-based stationary-point search. already-better.

### P3  One λ shared by all terms in the default gridsearch  — high — pyGAM-slop-to-avoid
- `pygam.py:~1966`: when `lam` is not per-term, gridsearch evaluates the same scalar for every term
  (11 candidates in total, not 11^p). A wiggly term and a linear/null term are forced to the same λ.
- Demo `d05_sharedlam.py` (n=400, y = sin(6πx) + linear z + null w), 5 reps (a 15-rep run
  timed out under load): the λ chosen was identical for all three terms in 5/5 reps (always λ=1.0, the
  same compromise for a wiggly term, a linear term and a null term). Mean in-sample MSE vs truth:
  pyGAM gridsearch 0.0151 vs gamfit REML 0.0073 (2.08× worse), and gamfit was better in 5/5 reps.
  Source: `pygam.py:1964–2016`. The default `param_grids["lam"] = np.logspace(-3,3,11)` is a scalar
  grid, and `combine(*grids)` sets that scalar on every term.
- gamfit: one ρ per penalty, jointly optimised by REML with analytic gradient/Hessian. already-better.

### P4  UBRE formula bug: `~add_scale` where `not add_scale` was intended  — low — bug (pyGAM only)
- `pygam.py:1208–1222`: `~True == -2` in Python, so the scale term is multiplied by −2/…; demo
  `d01_bugs.py`: UBRE 3.1321 vs intended 2.1321. Lesson: criteria must be unit-tested against closed
  forms. gamfit has no UBRE (good).

### P5  McFadden pseudo-R² inverted  — low — bug (pyGAM only)
- `pygam.py:1151` computes `ll/ll_null` instead of `1 − ll/ll_null`; demo: 0.7816 reported vs 0.2184
  correct. Also the null model at `pygam.py:1141` uses an *unweighted* mean. Do not ship pseudo-R²
  variants at all unless defined precisely (SPEC: delete unnecessary options).

### P6  "scale" stored as √φ for every family → wrong Gamma/InvGauss likelihood, sampling, AIC  — high — bug (pyGAM)
- `pygam.py:1033–1046` stores `scale = sqrt(phi)`; `distributions.py` Gamma/InvGauss `log_pdf` and
  `sample` then treat it as φ. Demo `d01_bugs.py`: true φ=0.5 → pyGAM scale 0.700 (=√0.5); simulated
  Gamma variance 2.80 vs correct 2.0; loglik −5681.9 vs correct −5593.7; InvGauss `sample` mean 0.75 vs 3.
  Every AIC/p-value for these families is therefore wrong.
- gamfit: Gamma loglik −5671.71 equals the correct-φ̂ likelihood (−5671.84) → estimate correct
  (see G3 for the reporting gap).

### P7  Prior-weight handling in log-likelihoods  — med — bug (pyGAM)
- `distributions.py` Normal `log_pdf` uses `scale/w` (variance vs sd confusion): w=5 gives −7.53 vs
  correct −2.23. Binomial `log_pdf` ignores weights entirely (identical −1.204 for w=1 and w=5).
  Poisson exposure (`pygam.py:2775–2890`) is implemented as y/E with weights×E and the loglik uses
  `round(y·w)`; with unit user weights it is exact (−3735.405 both ways, `d07`), and the round() only
  bites for non-unit user weights (not demonstrated → not claimed).
- gamfit: exposure goes in as an offset (`offset="logE"`); `d07`: rate RMSE 0.0665 vs pyGAM exposure 0.0858.
  Do not copy the y/E-with-weights trick; offsets are the principled parameterisation.

### P8  Smooth-term p-values: ad-hoc Wald on full-rank pinv, flagged "KNOWN BUG" by pyGAM itself  — med — pyGAM-slop-to-avoid
- `pygam.py:1280–1295`, docstring `1798–1806` ("KNOWN BUG: p-values ... too small"). It (i) centres
  the coefficients by their mean (not an identifiability constraint), (ii) uses a full-rank pseudo-
  inverse of the penalised covariance block, (iii) takes an F/χ² reference with `edof` df and no
  account for λ estimation.
- Null calibration, `d04_pcal.py` (n=200, y=sin(2πx)+N(0,.5²), test the pure-noise s(z)), 100 reps:

  | method                         | P(p<.05) | P(p<.01) | median p |
  |--------------------------------|----------|----------|----------|
  | pyGAM default λ=0.6            | 0.060    | 0.010    | 0.528    |
  | pyGAM gridsearch (GCV)         | 0.040    | 0.010    | 0.558    |
  | gamfit `p_value_corrected`     | 0.060    | 0.000    | 0.439    |
  | gamfit `p_value_conditional`   | **0.140**| 0.030    | 0.254    |

  Honest result: in this null design pyGAM's test is NOT demonstrably mis-sized (0.06 and 0.04 at the
  5% level; the Monte Carlo SE is about 0.022). The critique of P8 is therefore methodological, not
  empirical. The construction has no theory: coefficient mean-centring, a pinv of a rank-deficient
  block, and ref-df = edof. pyGAM's own docstring admits it is wrong, and it has no power/size guarantees
  once the basis or penalty changes. Do not copy it, but also do not claim here that it is anti-conservative.
- gamfit: `smooth_significance` = LR test with null-spectrum weighted-χ² reference plus a
  smoothing-corrected variant (see G4 for the API surface issue).

### P9  Intervals ignore smoothing-parameter uncertainty  — med — already-better
- `pygam.py:1402–1416` uses V_β = (XᵀWX+S_λ)⁻¹φ at λ̂ only; no Vc = Vβ + J Var(ρ̂) Jᵀ correction.
- gamfit: smoothing-corrected covariance is the default (`gamfit/_model.py:194–210`,
  `covariance_mode` default "smoothing-corrected" when available; `crates/gam-solve/src/rho_uncertainty.rs`).
- Checked and REFUTED: "pyGAM monotone-constrained CIs collapse" — they are wider (0.115 vs 0.070,
  `d02` (C)). Not reported as a finding.

### P10 `sample()` / `_bootstrap_samples_of_smoothing` is statistically broken  — med — bug (pyGAM)
- `pygam.py:2287–2337`: λ candidates `exp(randn(11,M)*6 − 3)` while the docstring says [1e-3,1e3];
  `d08_bootstrap.py`: 25.5% of draws < 1e-3, 4.9% > 1e3, 95% range [4.0e-7, 6.3e3].
  It runs gridsearch on a deepcopy of the *fitted* model, so the original-data incumbent competes with
  bootstrap fits: 19/40 bootstrap replicates return exactly the original coefficients. Also forces a
  progress bar. Do not copy; gamfit's ρ-uncertainty is derived from the REML Hessian (and, when used,
  its sampling is in `rho_uncertainty.rs`, see G6).

### P11 Shape constraints are soft 1e9 penalties on currently-violating differences  — high — already-better
- `pygam.py:187–189` (`_constraint_lam=1e9`, `_constraint_l2=1e-3`, `_constraint_l2_max=0.1`),
  `penalties.py` monotone masks only differences violating *at the current iterate*.
- `d03_monotone.py`: violation grows with n and with y-scale: max drop 9.3e-7 (n=300, scale 1) →
  1.03e-2 (scale 1e4) → 0.686 (n=20000, scale 1e4) → 69 (n=20000, scale 1e6). The constraint is not a
  constraint; its tightness depends on the data scale relative to a hard-coded 1e9.
- gamfit: exact derivative control-polygon cones (`crates/gam-terms/src/smooth/shape_constraints.rs:1–8,130`)
  solved by an active-set QP (`crates/gam-solve/src/active_set.rs`); `d06_gamfit_mono.py`: max drop
  exactly 0 at n=300 and at n=20000/scale 1e4.

### P12 Ridge (l2) on linear and factor terms — not scale-invariant  — med — pyGAM-slop-to-avoid
- `terms.py` LinearTerm/FactorTerm default `penalties='l2'`, `lam=0.6`. Demo `d02` (B): the fitted slope
  in original units for x rescaled by 1 / 0.01 / 0.001 is 0.5001 / 0.0368 / 0.0004 (OLS 0.5007). A unit
  change in a covariate changes the fit.
- The slop is the fixed, unselected λ=0.6 ridge on raw coefficients, which is not scale-invariant. The
  principled default still penalizes (a prior toward no effect, so the null is recoverable): a
  REML-selected, scale-invariant penalty on the function (sum-to-zero contrasts for factors), with only
  the intercept unpenalised. gamfit's unpenalised parametric terms are a gap against that default, not
  already-better.

### P13 No identifiability constraints  — med — already-better
- pyGAM builds each s() with a full B-spline basis plus an intercept; demo `d02` (I): s(0)+s(1)
  model matrix has 41 columns but rank 39; identification relies on the penalty + sqrt(EPS) ridge.
  Partial dependences are then only defined up to arbitrary constants.
- gamfit: sum-to-zero by default (`term_specs.rs:751–757`, `TensorBSplineIdentifiability::SumToZero`);
  y~s(x)+s(z) has 23 = 1+11+11 coefficients.

### P14 Non-convergence prints and returns a "fitted" model  — high — pyGAM-slop-to-avoid
- `pygam.py:756–822`: PIRLS loop exits at `max_iter` with `print('did not converge')` and sets
  `_is_fitted=True`. Demo `d02` (H): separated logistic data, max_iter=3 → prints, returns AIC 25.97.
  gridsearch (`~2023–2031`) silently skips `ValueError`s and returns self *unfitted* if all fail
  (warning only when verbose). SPEC: "a fit only comes from a converged optimisation".
- gamfit: SPEC.md:22 mandates "a fit object must only ever come from a converged optimization"; not independently stress-tested here (robustness auditor covers it).

### P15 Penalty on coefficients (D'D difference penalty)  — med — already-better
- `penalties.py` derivative penalty = 2nd differences of coefficients; with non-uniform knots or a
  change of basis this is not a penalty on f.
- gamfit: exact ∫(f^(m))² Gram (`crates/gam-terms/src/basis/derivative_penalty.rs:1–33,47`), used by
  `bspline_build.rs:399,614` and `position_basis.rs:460`. See G8 for a dead leftover.

### P16 Numerical jitter constants papering over the solver  — med — pyGAM-slop-to-avoid
- `pygam.py:749` adds sqrt(EPS)·I ridge on *all* coefficients (incl. intercept); `pygam.py:694` initial
  β = sqrt(EPS); `pygam.py:698–699` y±0.01 fudge for link init; `pygam.py:499–537` `_cholesky` retries
  with increasing diagonal loading. Each changes the estimator silently.
- Do not copy. gamfit equivalents to watch: see G6.

### P17 Expectile λ/τ found by bisection  — low — pyGAM-slop-to-avoid
- `pygam.py:3482–3551` `fit_quantile` bisects on the expectile level to hit an empirical quantile —
  derivative-free search with a tolerance knob; and still with the fixed/grid λ. Any quantile/expectile
  model in gamfit must be a proper loss with REML/LAML-selected smoothness.

### P18 Uniform knots over [min, max] of the training data  — low — pyGAM-slop-to-avoid
- `terms.py` / `utils.py` `b_spline_basis`: equally spaced knots on the observed range, with ad-hoc
  extrapolation; with skewed x most knots land in empty regions. Combined with a coefficient penalty,
  the implied smoothness prior depends on the x-distribution. gamfit's integrated-derivative penalty
  makes knot placement matter much less; keep it that way.

---------------------------------------------------------------------------------------------------

## Part B — slop inside gamfit (gaps & principled replacements)

### G1 Default knot cap (≤ 8 internal knots) is a magic constant hiding an optimiser stall; defaults underfit  — high — gap
- `crates/gam-terms/src/term_builder.rs:4477–4514`: `MAX_DEFAULT_INTERNAL_KNOTS = 8`,
  `heuristic_knots_for_column = (unique/4).clamp(4, 8)`; the comment cites gam#1680: with larger
  bases the double-penalty REML surface is flat and the outer optimiser stalls. Warning text at
  `term_builder.rs:2850`.
- Demo `d10_gamfit_kcap.py`, sin(8πx):

  | n    | default (12 coef) RMSE | k=30 RMSE | `basis_check` p |
  |------|------------------------|-----------|-----------------|
  | 500  | 0.1481                 | 0.0631    | 3.2e-9          |
  | 5000 | 0.1343                 | 0.0213    | 3.8e-130        |

  The default does not improve with n (inconsistent), the model's own `basis_check` rejects the basis
  and it warns "using 10.92 of 11 edf", but nothing acts on it. In `d09` this is the only row where
  pyGAM-gridsearch beats gamfit (0.0518 vs 0.1421). This is "papering over solver issues" (SPEC).
- Fix: (a) fix the flat-direction stall itself — when the null-space penalty's ρ runs to +∞ (flat REML
  in that coordinate) handle it by its analytic ρ→∞ limit (the term projected onto its null space), with stationarity
  certified on the reduced face, rather than capping k; (b) derive the default k from the data so the
  basis is rich enough that REML, not k, controls smoothness (the penalty makes excess k harmless), with
  no hand-picked constants or caps. `basis_check` stays a diagnostic; it does not drive a refit loop.
  Files: `crates/gam-terms/src/term_builder.rs` (delete the cap), `crates/gam-solve/src/rho_optimizer/*`
  (boundary handling for flat ρ), basis-adequacy code (the `basis_check` backend). Size: M–L.

### G2 Python model comparison uses an uncorrected conditional AIC computed in pyffi  — high — gap/bug
- `crates/gam-pyffi/src/model/model_ffi.rs:4308–4323` `ranking_score_from_summary_payload` =
  `−2·loglik + 2·edf` — statistics living in the FFI (SPEC: pyffi has no logic), and also called at
  `model_ffi.rs:1403–1409` and `crates/gam-pyffi/src/latent/reml_latent_fit_ffi.rs:4097–4105`.
  Consumers: `Model.conditional_aic` (repo `gamfit/_model.py:1257–1271`; installed `Model.evidence`),
  `evidence_ratio_vs`, `compare_models` (`gamfit/_compare.py:20`).
- Meanwhile Rust already computes the right thing: `crates/gam-inference/src/model_comparison.rs:8–17,447–450`
  (`aic_conditional` with scale dof, and the Wood–Pya–Säfken 2016 `aic_corrected` that adds the
  smoothing-parameter-uncertainty edf). Only consumer: `crates/gam-cli/src/main/run_diagnose.rs:169`.
  So CLI and Python disagree, and Python never exposes `aic_corrected` (grep for aic_corrected: no hit in gamfit/ or gam-pyffi).
- Demo `d11_gamfit_aic.py`: `evidence == −2ll + 2edf` exactly (268.58; Rust aic_conditional would be
  270.58). Over 20 reps with a pure-noise extra s(z), the conditional AIC prefers the over-fitted
  model in **9/20** reps — the known anti-conservatism of conditional AIC that the WPS correction exists
  to fix (Greven & Kneib 2010).
- Fix: carry `ModelComparison { aic_conditional, aic_corrected }` in the Rust summary payload; make
  Python `compare_models`/`evidence_ratio_vs` rank by `aic_corrected` and delete
  `ranking_score_from_summary_payload` from pyffi; keep one AIC in the public API (the corrected one).
  Files: `crates/gam-inference/src/model_comparison.rs`, `crates/gam-pyffi/src/model/model_ffi.rs`,
  `crates/gam-pyffi/src/latent/reml_latent_fit_ffi.rs`, `gamfit/_model.py`, `gamfit/_compare.py`. Size: S–M.

### G3 Estimated dispersion φ̂ not surfaced  — low — gap
- For Gamma (and Gaussian σ²) `summary().to_dict()` and the printed report have no scale/dispersion
  field; only the loglik reveals φ̂ is correct (−5671.71 vs −5671.84 at true φ). Users cannot report or
  check it. Fix: add `scale` (φ̂, with its REML definition) to the Rust summary payload and print it.
  Files: gam-inference summary builder, `model_ffi.rs` payload, `_model.py` summary. Size: S.

### G4 `smooth_significance` returns four p-values, one of them anti-conservative  — med — gap (SPEC "delete unnecessary options")
- Fields `p_value_conditional`, `p_value_bound`, `p_value_uncorrected`, `p_value_corrected`. A user must
  pick; three of the four are not the recommended test. Calibration (d04, 100 null reps): `p_value_corrected` has size 0.060 at 5% and 0.000 at 1% (conservative
  at 1%, which is a calibration bug to check with KS on the full range and size at several α with more reps). `p_value_conditional` has size **0.140** at 5% and 0.030 at 1%. That is
  anti-conservative, about 4 Monte Carlo SE above nominal (binomial P(X≥14 | 100, 0.05) ≈ 5e-4). An
  exposed field that nearly triples the type-I error is a trap, not an option.
  Fix: return one `p_value` (the calibrated, smoothing-corrected one) plus statistic/ref-df; delete the
  others. Files: gam-inference smooth test, pyffi payload,
  `gamfit/_model.py`. Size: S.

### G5 Magic seed lattices and screening budget in the outer optimiser  — med — gap
- `crates/gam-solve/src/seeding.rs:50–52` (`tau_anchors [primary,0,−2,2]`, `log_kappa_grid [−2..2]`,
  `nu_grid [1.25,1.5,2,2.5,3,4]`), local offsets ±0.3/±1/±0.5 at `89–91`, further lattices `119–120,157`;
  `crates/gam-solve/src/rho_optimizer/seed_screening.rs:143–180,285+` ranks lattice seeds by a
  capped-inner-iteration REML proxy with a seed budget of 2 and hard-coded multipliers.
- Not a violation of "REML only" (a certified derivative-based optimisation follows), but it is grid-
  like pre-selection with unexplained constants, and it is the kind of machinery that hides the G1
  stall. Fix: a single deterministic start derived from the problem itself (e.g. the Fellner–Schall
  fixed-point iterate, which is derivative-based and constant-free) followed by the Newton/trust-region
  REML; delete the lattices and the screening proxy. Files: `seeding.rs`, `rho_optimizer/seed_screening.rs`. Size: M.

### G6 Magic cost gates / tolerance floors in inference code  — low — gap
- `crates/gam-solve/src/rho_uncertainty.rs:14–16`: `DEFAULT_SAMPLE_COUNT=32`, `MAX_AUTO_RHO_DIM=4`,
  `MAX_AUTO_WORK_UNITS=2_000_000` — whether the user gets the smoothing-corrected covariance depends on
  a work budget; a 5-smooth model silently falls back. `crates/gam-solve/src/estimate/smoothing_correction.rs:14`
  `PIRLS_INNER_TOLERANCE_FLOOR=1e-6` and `.with_max_iterations(300)`.
- Fix: use the first-order analytic correction Vc = Vβ + J V_ρ Jᵀ (needs dβ/dρ, already available from
  the REML gradient machinery) for any dimension, no sampling and no budget; drop the gate. Size: M.

### G7 Stale docstring claims grid-generated constraint matrix  — low — bug (docs)
- `gamfit/smooth.py:101–121` says the constraint matrix is "generated from the basis on a dense
  1D grid"; the implementation is the exact control-polygon cone (`shape_constraints.rs:1–8`). Fix the
  docstring. Size: S.

### G8 Dead difference-penalty computation  — low — cleanup
- `create_difference_penalty_matrix` is only called in production by `duchon_thinplate.rs:3511` inside
  `compute_geometric_constraint_transform`; its callers `bspline_build.rs:1271,1436` destructure
  `let (z, _)` and discard the penalty. Remove the computation (and the coefficient-difference penalty
  from production paths). Size: S.

### G9 Default diagnostic spam; pandas input requires pyarrow  — low — gap
- Every fit writes `[OUTER]/[HGB]/[INDEF-HESS]` lines to stdout/stderr; a pandas DataFrame input raises
  ImportError without pyarrow (dict input works). Route diagnostics through `log` at debug level;
  accept DataFrames via `__array__`/column iteration without pyarrow. Size: S.

---------------------------------------------------------------------------------------------------

## Summary tables

"Do not copy" list: P1 fixed λ default; P2 GCV/UBRE, γ=1.4, λ grid; P3 shared λ; P6 √φ scale; P7 weight
handling / exposure as weights; P8 naive Wald p-values; P10 bootstrap-by-gridsearch; P11 soft 1e9
constraint penalties; P12 fixed, unselected ridge on raw parametric coefficients; P13 no identifiability; P14 non-converged fits
returned; P15 coefficient-difference penalty; P16 jitter constants; P17 bisection/derivative-free
search; P18 uniform range knots; P4/P5 unchecked formulas.

"Principled replacement" list (gamfit work): G1 remove knot cap, analytic ρ→∞ limit for flat ρ, data-derived k (basis_check a diagnostic); P12 REML-selected function penalty on parametric/factor terms;
G2 WPS-corrected AIC in Rust payload, delete pyffi scoring; G4 single calibrated p-value; G5 single
derivative-based start instead of seed lattices; G6 analytic Vc without cost gates; G3 expose φ̂;
G7/G8/G9 cleanup.
