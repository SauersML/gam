# Audit axis: response families and link functions

gamfit 0.1.267 wheel (repo HEAD d10950d, pyproject 0.1.268) vs pyGAM 0.12.0.
Scripts and raw outputs: `scratchpad/audit/families/`
(`probe_api.py`, `compare_fits.py`, `probe_sas.py`, `probe_pygam_binom.py`,
`probe_pygam_cells.py`, `probe_pygam_exposure_gs.py`, `probe_expectile.py`,
`probe_ig_proxy.py`, `probe_expectile2.py`, raw outputs `compare_all.txt`,
`expectile_out.txt`, `expectile2_out.txt`).
Python was run from the scratchpad (running from `/home/user/gam` imports the
source tree, which has no `_rust`). gamfit was fed dict-of-numpy input
(pandas input fails without pyarrow: `ImportError: Import pyarrow failed`,
not this axis). No `gam` CLI binary is installed, so CLI parity is judged
from source (`crates/gam-cli`).

## 1. Coverage matrix (pyGAM feature -> gamfit)

| pyGAM feature | gamfit Python | formula DSL | CLI | Works in practice? |
|---|---|---|---|---|
| LinearGAM (normal/identity) | `family="gaussian"` | - | `--family gaussian` | yes, better than pyGAM (table 2) |
| LogisticGAM (binomial/logit) | `family="binomial"` | `link(type=logit)` | `--family binomial-logit` | yes, better |
| binomial with trials/levels | proportions `y` + `weights=` | - | `--weights` | yes, better (pyGAM per-row levels is broken, see F8) |
| PoissonGAM | `family="poisson"` | - | `--family poisson-log` | yes |
| PoissonGAM exposure | `offset="log_e_column"` | `offset(...)` | `--offset` | yes, better (pyGAM gridsearch+exposure is broken, F7) |
| GammaGAM (default link log) | `family="gamma"` (log) | - | `--family gamma-log` | yes (table 2) |
| GAM(gamma, link="inverse") (canonical) | NO: `unsupported link type 'inverse'` | NO | NO | gap F2 |
| InvGaussGAM | NO: `unknown family 'inverse-gaussian'` | NO | NO | gap F1 |
| ExpectileGAM | `family="expectile"`, `expectile_tau=` or `expectile(0.9)` | - | `--family expectile --expectile-tau` | yes (LAWS + REML), see F5 |
| ExpectileGAM.fit_quantile | NO (by design; CTN quantiles exist) | - | - | pyGAM slop, F6 |
| GAM(normal, link=log / inverse / inv_squared) | NO | NO | NO | gap F2 |
| GAM(poisson, identity) | NO (rejected by legality matrix) | - | - | gap F2 (pyGAM accepts it unconstrained, F9) |
| GAM(gamma, identity) | NO | - | - | gap F2 |
| GAM(binomial, identity) | NO | - | - | pyGAM accepts silently, F9; gamfit rejection is correct |
| sample weights | `weights=` | - | `--weights` | yes (pyGAM casts to float32, F7) |
| not in pyGAM: probit, cloglog, loglog, cauchit, beta-logistic, latent-cloglog, link mixtures, SAS | yes | yes | yes | yes except SAS (bug F4) |
| not in pyGAM: negative binomial, beta, tweedie, multinomial, Royston-Parmar survival, transformation-normal | yes | - | yes | already-better F10 |

Evidence of the pinned-link legality matrix:
`crates/gam-spec/src/lib.rs:1457-1491` (`is_legal_cell`: Gaussian/Royston-Parmar
only Identity; Poisson/Gamma/Tweedie/NB only Log; Beta only Logit; Binomial all
probability links but not Identity/Log), mirrored in
`crates/gam-models/src/fit_orchestration/materialize/family.rs:56-76`
(`link_legal_for_family`). `LinkFunction` at `crates/gam-spec/src/lib.rs:116-126`
has no Inverse/InverseSquared/Sqrt variant; `ResponseFamily` at
`crates/gam-spec/src/lib.rs:605-631` has no InverseGaussian.

`probe_api.py` output (abridged):
```
FAIL gauss {'family': 'gaussian', 'link': 'log'} -> link 'log' is not supported for family 'gaussian'
FAIL gauss {'family': 'gaussian', 'link': 'inverse'} -> unsupported link type 'inverse'; use one of identity|log|logit|probit|cloglog|loglog|cauchit|...
FAIL pos   {'family': 'gamma', 'link': 'inverse'} -> unsupported link type 'inverse'
FAIL pos   {'family': 'gamma', 'link': 'identity'} -> not supported
FAIL pos   {'family': 'gamma(inverse)'} -> ... use one of identity|log|logit|probit|cloglog|sas|beta-logistic   (vocabulary differs from the one above)
FAIL pos   {'family': 'inverse-gaussian' | 'inverse_gaussian' | 'inv_gauss' | 'invgauss'} -> unknown family
FAIL count {'family': 'poisson', 'link': 'identity'} -> not supported
FAIL count {'family': 'poisson', 'link': 'sqrt'} -> unsupported link type
OK   bin   binomial logit / probit / cloglog / loglog / cauchit, binomial(cauchit)
FAIL bin   {'family': 'binomial', 'link': 'log'} -> relative-risk link not supported
FAIL bin   {'family': 'binomial', 'link': 'sas'} -> (see F4)
OK   gauss expectile, expectile_tau=0.9  and  family='expectile(0.9)'   (family_name reports 'Gaussian Identity')
FAIL gauss {'family': 'quantile'} -> unknown family
```

## 2. Head-to-head fit quality

Truth `eta = sin(2 pi x1) + 0.8*4(x2-0.5)^2 - 0.27`, formula `y ~ s(x1) + s(x2)`,
pyGAM `s(0) + s(1)`. RMSE of the fitted mean on the response scale against the
known truth on a 60x9 held-out grid; `cover95` = fraction of grid points whose
95% interval for the mean contains the truth. pyGAM default = fixed lam=0.6;
pyGAM gridsearch = its recommended 11-point lam grid on GCV/UBRE. The machine
was heavily oversubscribed (load average 40+ on 4 CPUs), so absolute times
are inflated; only ratios within a row block are meaningful.

| scenario | n, reps | gamfit RMSE / cover95 / time | pyGAM default | pyGAM gridsearch |
|---|---|---|---|---|
| gaussian, sd 0.5 | 600, 8 | **0.0652** / 0.976 / 0.18s | 0.0914 / 0.960 / 0.02s | 0.0695 / 0.911 / 0.16s |
| binomial (Bernoulli, logit) | 1500, 8 | 0.0329 / **0.962** / 3.8s | 0.0552 / 0.941 / 1.3s | **0.0321** / 0.936 / 8.8s |
| binomial trials 1..20 (prop + weights) | 600, 8 | **0.0177** / 0.973 / 4.7s | 0.0310 / 0.941 / 8.7s | 0.0204 / 0.879 / 67.9s |
| poisson with exposure U(0.2,5) (gamfit offset=log e) | 600, 8 | **0.0751** / 0.989 / 5.0s | 0.1440 / 0.954 / 20.1s | 0.7093 / **0.000** / 116s (pyGAM bug F7) |
| gamma, log link, shape 3 | 600, 4 | **0.2118** / 0.937 / 14.3s | 0.3515 / 0.938 / 24.6s | 0.2813 / 0.859 / 229s |
| gamma, truth on inverse link (gamfit: gamma(log), misspecified control) | 600, 4 | **0.1317** / 0.932 / 8.8s | 7.9636 / 0.016 / 29.9s (GAM(gamma, link="inverse")) | all 4 reps crash: `AttributeError: 'NoneType' object has no attribute 'get_params'` |
| inverse Gaussian, log link, lambda 10 | 600, 2 | unsupported (`unknown family`); gamma(log) proxy: 0.1218 / 1.000 and 0.2203 / 0.956 (probe_ig_proxy.py, same seeds) | 0.2902 / 0.965 / 20.2s | 0.1821 / 0.955 / 209s |
| expectile tau=0.9, heteroscedastic sd 0.2+0.8 x1 | 600, 4 | **0.1158** / 0.892 / 1.4s | 0.1484 / 1.000 / 20.0s | 0.1170 / 0.994 / 129s |

Expectile deep-dive (`probe_expectile.py 3`, y = sin(2 pi x) + (0.2+0.8x) e, n=800, 3 reps,
100-point grid, `y ~ s(x)` vs `ExpectileGAM(s(0)).gridsearch`):

| tau | gamfit RMSE / cover95 / time | pyGAM gridsearch RMSE / cover95 / time |
|---|---|---|
| 0.05 | 0.0859 / 0.807 / 1.2s | 0.0819 / 1.000 / 38.9s |
| 0.50 | 0.0617 / 0.947 / 0.2s | 0.0564 / 0.960 / 25.2s |
| 0.95 | 0.1007 / 0.727 / 1.7s | 0.0902 / 1.000 / 41.8s |

Crossings of the three separately fitted expectile curves on the grid: 0 of 300
for both libraries (this design is not adversarial for crossing).

Paired 8-seed follow-up (`probe_expectile2.py 8`, n=800, 1-D, seeds 700-707):
tau=0.5 gamfit 0.0496 vs pyGAM gridsearch 0.0455 (gamfit wins 3/8), coverage 0.971;
tau=0.95 gamfit 0.0772 vs 0.0771 (wins 4/8), coverage 0.904.

Reading: for the GLM families gamfit supports (gaussian, binomial, binomial
trials, poisson+offset, gamma-log), gamfit's REML/LAML fit is at least as
accurate as pyGAM's best (gridsearch, which costs 2-20x more time) and its
intervals are calibrated or slightly conservative (0.937-0.989), whereas pyGAM
gridsearch intervals under-cover (0.859-0.936). pyGAM default (lam=0.6) is
uniformly less accurate. Where pyGAM has a feature gamfit lacks, pyGAM's
implementation is itself broken in two of three cases (gamma inverse link;
gridsearch with exposure); inverse Gaussian is the one real, working pyGAM
capability gamfit lacks. For expectiles gamfit ties pyGAM gridsearch in
accuracy (not strict domination) and its bands under-cover at extreme tau.

## 3. Findings

### F1. No inverse-Gaussian family
- kind: gap. severity: high.
- evidence: `gamfit.fit(d, "y ~ s(x)", family="inverse-gaussian")` (also
  `inverse_gaussian`, `inv_gauss`, `invgauss`) -> `unknown family`
  (probe_api.py). `ResponseFamily` enum `crates/gam-spec/src/lib.rs:605-631`
  has no such variant; CLI `FamilyArg` `crates/gam-cli/src/main/cli_args.rs:595-616`
  has none. pyGAM `InvGaussGAM` fits (RMSE 0.182 gridsearch). A gamma(log)
  proxy in gamfit is as accurate for the mean (0.171 average, same seeds) but
  the variance function (mu^2 vs mu^3), dispersion, deviance, AIC/LAML value,
  observation intervals and posterior-predictive sampler are all wrong for IG
  data, so the proxy is not a substitute.
- proposed fix (beyond pyGAM): `ResponseFamily::InverseGaussian` with
  V(mu)=mu^3, log-likelihood `-1/2 log(2 pi phi y^3) - (y-mu)^2/(2 phi y mu^2)`,
  dispersion phi estimated as an outer LAML hyperparameter with an analytic
  gradient exactly as the Gamma shape is (never profiled by search, never GCV).
  Links: log (default; always feasible) and canonical 1/mu^2 (feasibility
  eta>0 handled by the existing recoverable step rejection,
  `PirlsRowGeometryUnrepresentable`, never by clamping). Write the row NLL once
  as a `RowProgram` (`crates/gam-math/src/jet_tower.rs:1294`) so the exact
  fourth-order tower needed by LAML derivatives is generated mechanically
  (the module doc at jet_tower.rs:102 already names this as the path for new
  families), then feed it through the GLM PIRLS/LAML path. Observation law:
  Wald sampler + exact quantiles in `gam-predict`. pyGAM uses the Fisher
  scoring working weights and GCV; we get exact LAML and calibrated
  predictive intervals.
- files: crates/gam-spec/src/lib.rs (enum, legality, names),
  crates/gam-models/src/fit_orchestration/materialize/family.rs (name
  resolver, default link), crates/gam-solve/src/pirls/{working_model_trait.rs,
  log_link_working_state.rs, deviance.rs, dispersion.rs, curvature.rs},
  crates/gam-solve/src/gpu_kernels/pirls_row.rs (or explicit CPU-only
  routing), crates/gam-models/src/family_runtime.rs,
  crates/gam-models/src/inference/model.rs (persistence),
  crates/gam-predict/src/{lib.rs, generative.rs},
  crates/gam-inference/src/hmc_io.rs, crates/gam-cli/src/main/{cli_args.rs,
  family_resolve.rs}, gamfit/_api.py docstring, docs/families-and-links.md.
  (~26 non-test files reference `ResponseFamily::Gamma`; IG touches the same set.)
- size: M-L.

### F2. Non-binomial families are pinned to one link; no inverse / inverse-squared / sqrt links
- kind: gap. severity: med.
- evidence: probe_api.py: gaussian+log, gaussian+inverse, gamma+inverse,
  gamma+identity, `gamma(inverse)`, poisson+identity, poisson+sqrt all
  rejected. `is_legal_cell` `crates/gam-spec/src/lib.rs:1457-1491`;
  `link_legal_for_family` `materialize/family.rs:56-76`; `LinkFunction`
  `crates/gam-spec/src/lib.rs:116-126` has no Inverse/InverseSquared/Sqrt;
  PIRLS kernels are hard-coded per cell (`working_model_trait.rs:346-520`,
  `glm_update.rs:121-150`: e.g. Poisson always uses
  `write_poisson_log_working_state`). pyGAM exposes identity/log/logit/
  inverse/inv_squared for any distribution (pygam/links.py:317).
  Severity is med, not high, because pyGAM's own non-canonical cells are
  broken in practice: GAM(gamma, link="inverse") on data generated on the
  inverse link gives RMSE 7.96 (coverage 0.016) at default lam and every
  gridsearch rep crashes (`AttributeError: 'NoneType' object has no attribute
  'get_params'`), while gamfit's misspecified gamma(log) fit gets RMSE 0.132 /
  coverage 0.932 (compare_fits.py gamma_inverse). The gap is expressiveness
  (identity-link Poisson for additive risk, log-link Gaussian, canonical
  gamma/IG), not accuracy on today's benchmarks.
- proposed fix: one generic exponential-dispersion row kernel, composed from a
  variance-function jet (V, V', V'', V''') and an inverse-link jet (mu, mu',
  mu'', mu''', mu''''), written once as a `RowProgram` so observed (Newton)
  information and the third/fourth derivatives LAML needs are exact; the
  hand-written canonical kernels stay as fast paths and become oracle tests
  for the generic one (jet_gamma_oracle_tests.rs / jet_poisson_oracle_tests.rs
  already exist in gam-math). Add `LinkFunction::{Inverse, InverseSquared,
  Sqrt}`. Replace the pin-one-link legality matrix with a support-based one:
  a link is legal for a family when its range can land in the family's mean
  domain; links whose range exceeds the domain (identity/inverse for
  Poisson/Gamma, 1/mu^2 for IG, identity for Gaussian-log style constraints)
  are legal but carry a feasibility set, enforced by recoverable step
  rejection at PIRLS/ARC trial points, with a typed error if the converged
  optimum sits on the boundary (no clamping, no silent pyGAM-style acceptance).
  Initial eta must be feasible by construction (e.g. from the log-link fit
  mapped through the target link). Do not add identity/log for binomial except
  as in F3.
- files: crates/gam-spec/src/lib.rs, crates/gam-terms/src/inference/formula_dsl.rs
  (parse_linkname ~L3472), crates/gam-models/src/fit_orchestration/materialize/family.rs,
  crates/gam-solve/src/pirls/{working_model_trait.rs, glm_update.rs,
  family_state.rs, deviance.rs, curvature.rs}, crates/gam-math/src/jet_tower.rs
  (new row programs), crates/gam-solve/src/gpu_kernels/pirls_row.rs (route
  non-canonical cells to CPU explicitly), crates/gam-predict/src/lib.rs,
  crates/gam-inference/src/hmc_io.rs, docs/families-and-links.md.
- size: L.

### F3. Binomial log link (relative-risk regression) rejected
- kind: gap. severity: low.
- evidence: `gamfit.fit(bin, "y ~ s(x)", family="binomial", link="log")` ->
  relative-risk link not supported; `is_legal_cell` lib.rs:1484 returns false
  for Binomial+Log. pyGAM's binomial+log also fails (OptimizationError,
  probe_pygam_cells.py), so this is not a parity loss, but log-binomial
  additive models are a standard epidemiology request.
- proposed fix: falls out of F2's feasibility machinery: eta<0 feasibility set
  enforced by recoverable step rejection; typed boundary error if the optimum
  hits mu=1. No clamped "log-binomial approximations" (Poisson-with-robust-SE
  trick is a different estimator and must not masquerade as this link).
- files: same as F2 plus crates/gam-solve/src/pirls/family_state.rs.
- size: S once F2 exists (M alone).

### F4. Binomial SAS link fails on ordinary logistic data (fatal outer ARC error)
- kind: bug. severity: high.
- evidence: `probe_sas.py` (n=1500, x~U(0,1), eta=sin(2 pi x), Bernoulli):
  seeds 0, 1, 2 all fail with `IntegrationError: Fatal outer-objective
  evaluation failure (outer ARC evaluation): ... PIRLS row geometry is not
  representable at row 0: saturated Bernoulli row inconsistent with response
  evaluated from eta=-10240.0 produced 0.0` (seed 1: eta=10240.0; seed 2:
  row 5). Same data with `link="beta-logistic"` and `"logit"` fit fine (RMSE
  0.022-0.033). Error text originates at crates/gam-solve/src/pirls/family_state.rs:224.
  HEAD carries a fix attempt: crates/gam-solve/src/rho_optimizer/run_plan_tests.rs:2829-2860
  (`outer_second_order_bridge_rejects_a_candidate_whose_row_geometry_refuses_2627`,
  "the payload both SAS-link CLI fits abort with, eta=-10240.0") makes that
  refusal recoverable at ARC trial points. Whether 0.1.267 contains it cannot
  be determined (shallow clone, no cargo build allowed); the wheel still
  reports it as Fatal.
- proposed fix: (1) keep the #2627 bridge change; (2) add a Python-level
  regression test that runs probe_sas.py's fit and requires a converged,
  certified fit; (3) root cause: an ARC trial reaching |eta|=10240 means the
  SAS (epsilon, log delta) step is unbounded in the tail; the SAS
  parameterization should keep delta on the log scale with the LAML
  curvature supplying the trust region (no hand bounds), and the saturated
  row should be expressed with log-space Bernoulli terms (log Phi / log1p
  forms) so a trial point is evaluable rather than refused. If the SAS
  parameters are not identifiable on logistic data (delta->1, eps->0 is the
  logit-like interior point), the outer problem should converge there, not
  wander to saturation.
- files: crates/gam-solve/src/rho_optimizer/bridges.rs (~1500-1560),
  crates/gam-solve/src/pirls/family_state.rs:224,
  crates/gam-solve/src/pirls/sas_saturated_row_2733_tests.rs, the SAS link
  state in gam-spec/gam-math, tests/ (python), crates/gam-cli/tests/suite/bug_hunt_sas_link_*.rs.
- size: S-M.

### F5. Expectile regression: supported and REML-principled, but its intervals under-cover
- kind: gap (uncertainty) on top of already-better (fit). severity: med.
- evidence (already-better part): `family="expectile", expectile_tau=0.9` and
  `family="expectile(0.9)"` both work in Python; CLI has `--family expectile
  --expectile-tau` (cli_args.rs:370-374, 595-616); shared request field
  `expectile_tau` (crates/gam-config/src/fit_request_document.rs:76-121).
  Implementation: crates/gam-models/src/fit_orchestration/entry.rs:2156-2375
  (`fit_expectile_laws`): Newey-Powell LAWS where every inner solve is a full
  Gaussian-identity GAM with REML lambda selection on the asymmetric weights,
  convergence certified by a dimensionless KKT residual of the original
  asymmetric objective (entry.rs:326-392), sign-cycle detection returns a typed
  error (entry.rs:287-320). This is the LAML stationary point of the
  asymmetric-normal working likelihood (at fixed signs, -1/2 sum log w_i does
  not depend on (lambda, sigma), and H depends on beta only through the
  piecewise-constant signs), so it is REML-principled, unlike pyGAM's fixed
  lam / GCV gridsearch. Accuracy: tau=0.9 2-D design RMSE 0.1158 vs pyGAM
  gridsearch 0.1170 and default 0.1484, at 1.4 s vs 129 s.
  Paired 8-seed 1-D heteroscedastic check (probe_expectile2.py 8, n=800):
  tau=0.95 RMSE 0.0772 vs pyGAM gridsearch 0.0771 (gamfit wins 4/8), tau=0.5
  0.0496 vs 0.0455 (wins 3/8); the 3-seed probe_expectile.py run had pyGAM
  gridsearch ahead by 5-10% at all three tau. So on accuracy gamfit ties
  pyGAM's best at 20-100x lower cost, but does NOT strictly dominate it; the
  tau=0.5 deficit is plain Gaussian mean regression under heteroscedastic
  noise (REML with a constant-variance working model vs GCV), which belongs to
  the accuracy axis; the principled remedy there is gamfit's existing Gaussian
  location-scale family, not GCV.
- evidence (gap part): nominal-95% bands cover the true expectile 0.892
  (tau=0.9, 2-D), 0.807 (tau=0.05), 0.727 (tau=0.95), 0.947 (tau=0.5)
  (compare_fits.py expectile90, probe_expectile.py). entry.rs:2152-2155 states
  it: "expectile standard errors are the sandwich-free Gaussian-form bands of
  the converged weighted problem (a deliberate first-rung choice; see #1100)".
  The asymmetric-normal likelihood is a working likelihood, not the data law,
  so its Bayesian covariance is miscalibrated exactly where the weights are
  most asymmetric. (pyGAM's bands over-cover at 0.994-1.000: also
  miscalibrated, in the other direction.)
- proposed fix (beyond pyGAM):
  (a) Report a penalized sandwich (Huber-White / Newey-Powell) covariance
  for the expectile coefficients: V = H^-1 (X' diag(w_i^2 r_i^2) X) H^-1
  with H = X'WX + S_lambda at the certified fixed point; make it the default
  posterior covariance for expectile fits (the mean stays the converged
  estimator; no knob). Validate by coverage simulation in CI (tests, no XFAIL).
  (b) Joint multi-tau fit with a non-crossing guarantee: either expectile
  sheets (a smooth surface in (x, tau) with a penalty on the tau direction,
  Schnabel and Eilers 2009) or a location-scale expectile model
  mu(x) + sigma(x) e_tau reusing the existing gaussian-location-scale machinery,
  so e.g. `expectile_tau=[0.1, 0.5, 0.9]` returns one coherent fit instead of
  separately fitted, possibly crossing curves (no crossings observed in the
  probe, but nothing prevents them).
  (c) Minor: `model.family_name` reports "Gaussian Identity" for an expectile
  fit (probe_api.py); the saved model does carry the expectile tag
  (crates/gam-models/src/inference/model.rs:5099-5130), so surface
  "Expectile(tau=0.9)" in family_name/summary. docs/families-and-links.md does
  not mention expectile at all (it is only in docs/cli.md, diagnostics.md,
  frontend-parity.md, posterior-sampling.md).
- files: crates/gam-models/src/fit_orchestration/entry.rs (fit_expectile_laws,
  covariance at the fixed point), crates/gam-solve/src/model_types/result_types.rs
  (covariance slot), crates/gam-models/src/inference/model.rs (family label),
  crates/gam-predict/src/lib.rs (bands), crates/gam-pyffi/src/model/model_ffi.rs,
  docs/families-and-links.md; for (b) additionally crates/gam-config
  (vector tau), crates/gam-cli/src/main/cli_args.rs (--expectile-tau list).
- size: (a) M, (b) L, (c) S.

### F6. pyGAM's fit_quantile (bisection over expectiles on in-sample coverage) must not be copied
- kind: pyGAM-slop-to-avoid. severity: med.
- evidence: pygam.py:3482 `fit_quantile(self, X, y, quantile, max_iter=20,
  tol=0.01)`: bisects the expectile level until the in-sample fraction of
  y below the curve hits the target; that is a derivative-free search with a
  hand-chosen tolerance and iteration cap, and it re-uses in-sample coverage
  as the objective (optimistic, not a quantile estimator). `family="quantile"`
  is rejected by gamfit (probe_api.py), which is correct.
- proposed fix: none to the engine; for conditional quantiles point users to
  the transformation-normal family, which already yields non-crossing
  conditional quantiles analytically (crates/gam-predict/src/transformation_normal.rs,
  crates/gam-models/src/transformation_normal/quantile_table.rs). If a direct
  quantile-GAM is ever wanted, the principled route is a smoothed check loss /
  asymmetric-Laplace working likelihood with LAML-selected smoothing and a
  calibrated (e.g. Fasiolo et al. 2021 "qgam") learning rate derived by
  REML-type criteria, never bisection. Document the CTN route in
  docs/families-and-links.md.
- files: docs/families-and-links.md (S); engine none.
- size: S.

### F7. pyGAM PoissonGAM exposure handling: gridsearch double-applies exposure; float32 casts
- kind: pyGAM-slop-to-avoid (gamfit already-better). severity: high (for pyGAM users) / none for gamfit.
- evidence: `probe_pygam_exposure_gs.py`: same data, same lam=1:
  `PoissonGAM(lam=1).fit(X, y, exposure=e)` RMSE 0.0877, mean rate 1.031
  (truth 1.069); `PoissonGAM().gridsearch(X, y, exposure=e, lam=[1,1])`
  RMSE 0.7033, mean rate 0.405. Cause: PoissonGAM.gridsearch divides y by
  exposure and sets weights*=exposure (pygam.py:3037), then the base
  gridsearch calls `gam.fit(X, y, weights)` positionally (pygam.py:2054),
  which lands in PoissonGAM.fit's `exposure` slot and normalizes again. In
  the 8-rep benchmark pyGAM gridsearch+exposure has RMSE 0.709 and coverage
  0.000. Also exposure and weights are cast to float32 (pygam.py:2862, 2877,
  2954, and weights at 355, 889, 978, 1935, 2827). gamfit's `offset=` path:
  RMSE 0.0751, coverage 0.989 vs pyGAM default 0.1440 / 0.954.
- proposed fix: none needed; add a gamfit regression test that an offset of
  log(e) is equivalent to rescaled rate data (and a pyGAM-oracle test that
  documents the pyGAM bug so nobody "matches" it). Keep all inputs float64.
- files: tests (python) only.
- size: S.

### F8. pyGAM binomial `levels` (trials): per-row levels crash, scalar levels silently wrong
- kind: pyGAM-slop-to-avoid (gamfit already-better). severity: med.
- evidence: `probe_pygam_binom.py`: `GAM(distribution=BinomialDist(levels=trials_array))`
  -> `AssertionError: transformed response values should be well-behaved`;
  scalar `levels=20` with trials varying 1..20 runs and returns RMSE 0.2414
  (wrong model, no warning); `LogisticGAM(prop, weights=trials)` is the only
  correct pyGAM route (RMSE 0.0121). gamfit accepts proportions + `weights=`
  (benchmark: RMSE 0.0177 / coverage 0.973 vs pyGAM gridsearch 0.0204 / 0.879).
  gamfit has no explicit trials syntax (no cbind/trials in formula_dsl.rs or
  _api.py).
- proposed fix (optional, low): an explicit `trials=` column (or
  `cbind(successes, failures)` formula response) that the Rust validator
  checks (integer successes <= trials), because `weights` conflates trials
  with importance/frequency weights; it would also let observation intervals
  and posterior-predictive samples be Binomial(trials, p) instead of
  Bernoulli. Must be one spelling across CLI/Python/Rust.
- files: crates/gam-terms/src/inference/formula_dsl.rs, crates/gam-config/src/fit_request_document.rs,
  crates/gam-models/src/fit_orchestration/materialize/family.rs (validator),
  crates/gam-predict/src/generative.rs, crates/gam-cli/src/main/cli_args.rs, gamfit/_api.py.
- size: M.

### F9. pyGAM silently accepts incoherent (distribution, link) cells
- kind: pyGAM-slop-to-avoid. severity: med.
- evidence: `probe_pygam_cells.py`: pyGAM fits normal+inv_squared,
  poisson+identity, gamma+identity and binomial+identity (mu 0.12-0.46,
  edof 16) with no feasibility handling; poisson+logit is only rejected by
  accident (`AttributeError: 'PoissonDist' object has no attribute 'levels'`),
  binomial+log diverges (OptimizationError). gamfit's legality matrix
  (lib.rs:1457-1491) rejects every illegal cell with a precise message, which
  is already better.
- proposed fix: when F2 opens non-canonical cells, open them with an explicit
  feasibility set and a typed boundary error, never pyGAM's unconstrained
  acceptance; binomial+identity stays rejected (a linear probability GAM is
  a Gaussian model and should be spelled as one).
- files: crates/gam-spec/src/lib.rs (with F2).
- size: S (as part of F2).

### F10. Families and links where gamfit already dominates pyGAM
- kind: already-better. severity: low (informational).
- evidence: compare_fits.py (section 2): gaussian 0.0652 vs 0.0695
  (gridsearch), coverage 0.976 vs 0.911; binomial 0.0329 vs 0.0321 (tie),
  coverage 0.962 vs 0.936 and 2.3x faster than gridsearch; binomial trials
  0.0177 vs 0.0204, coverage 0.973 vs 0.879; poisson+exposure 0.0751 vs 0.1440
  (pyGAM gridsearch broken, F7); gamma(log) 0.2118 vs 0.2813, coverage 0.937
  vs 0.859. Links absent from pyGAM that gamfit fits (probe_api.py / probe_sas.py):
  probit, cloglog, loglog, cauchit, beta-logistic, latent-cloglog, mixture;
  families absent from pyGAM: negative binomial, beta, tweedie (explicit p),
  multinomial, Royston-Parmar, transformation-normal (materialize/family.rs
  ~340-540).
- proposed fix: pin these as regression benchmarks (accuracy and coverage
  against pyGAM) so they stay better.
- files: tests / bench only.
- size: S.

### F11. `link="log"` without a family guesses Poisson vs Gamma from integer-ness of y
- kind: gap (spec hygiene). severity: low.
- evidence: crates/gam-models/src/fit_orchestration/materialize/family.rs:617-629:
  `LinkFunction::Log => if y.iter().all(|&yi| yi.is_finite() && yi >= 0.0 && yi == yi.round()) { Poisson } else { Gamma }`.
  A positive continuous response that happens to be rounded (e.g. costs in
  whole dollars) becomes a Poisson model, a different variance function,
  from a data coincidence.
- proposed fix: with an explicit link and no family, require the family (typed
  error naming poisson/gamma/tweedie/negative-binomial), or pick by the
  documented `family="auto"` rule only; delete the implicit branch.
- files: crates/gam-models/src/fit_orchestration/materialize/family.rs, docs.
- size: S.

### F12. Inconsistent link vocabulary in error messages
- kind: bug (UX). severity: low.
- evidence: `family="gamma(inverse)"` lists `identity|log|logit|probit|cloglog|sas|beta-logistic`
  (materialize/family.rs:108) while `link="inverse"` lists
  `identity|log|logit|probit|cloglog|loglog|cauchit|binomial-logit|...|blended(...)/mixture(...)|flexible(...)`
  (crates/gam-terms/src/inference/formula_dsl.rs:3472). loglog/cauchit are
  valid but missing from the first.
- proposed fix: generate both messages from the single `parse_linkname`
  vocabulary table.
- files: materialize/family.rs, formula_dsl.rs.
- size: S.


### F13. Link spelled two ways in Python, one way in the CLI
- kind: gap (parity / option hygiene). severity: low.
- evidence: Python `gamfit.fit(..., link=...)` (gamfit/_api.py:668-712) and
  the formula term `link(type=...)` (crates/gam-terms/src/inference/formula_dsl.rs:3219)
  both set the link; the CLI deliberately has only the formula term
  (docs/cli.md:47-48 "there is no top-level `gam fit --link` flag"), while
  the shared request document still carries a `link` field
  (crates/gam-config/src/fit_request_document.rs:76-121). The CLI `FamilyArg`
  also has no binomial-loglog / binomial-cauchit spellings although Python's
  `binomial(cauchit)` works (cli_args.rs:595-616).
- proposed fix: pick one canonical spelling across frontends (either add
  `--link` to the CLI or delete Python's `link=` in favour of `family(link)`
  / `link(type=...)`), per SPEC's parity and delete-unnecessary-options rules.
- files: gamfit/_api.py, crates/gam-cli/src/main/{cli_args.rs, family_resolve.rs},
  crates/gam-config/src/fit_request_document.rs, docs/cli.md.
- size: S.

## 4. Suggested order of work
1. F4 (SAS fatal error; verify #2627 fix in a released wheel, add Python regression test). S-M.
2. F5a (sandwich covariance for expectiles; coverage test). M.
3. F1 (inverse Gaussian via a RowProgram row NLL, log link default). M-L.
4. F2 + F3 + F9 (generic variance x link jet kernel, support-based legality, feasibility sets). L.
5. F5b (joint non-crossing multi-tau expectiles), F8 (explicit trials). L / M.
6. F11, F12, F13, F5c, F6 docs. S each.
