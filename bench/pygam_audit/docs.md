# Audit: documentation, visualization and onboarding (pyGAM user switching to gamfit)

Scope: README.md, README_PYPI.md, docs/getting-started.md, docs/sklearn.md, docs/cookbook.md,
docs/predictions.md, docs/diagnostics.md, docs/formulas.md, docs/api-reference.md, mkdocs.yml,
wheel gamfit 0.1.267 (installed in bvenv) cross-checked against HEAD source (no builds).
Scripts and images: scratchpad/audit/docs/ (t0..t4b *.py, *.png, *.out, *.err). Data pulled from
Rdatasets because pyGAM 0.12.0's wheel ships no dataset CSVs (`pygam.datasets.wage()` raises
FileNotFoundError), so the pyGAM tour itself does not run from `pip install pygam`.

Machine was heavily loaded (load average 35-42 on 4 CPUs) throughout; absolute timings are inflated
for both libraries, ratios are indicative only.

## Tour reproduction summary

| Tour example | pyGAM | gamfit | Notes |
|---|---|---|---|
| Wage `s(year)+s(age)+f(education)` PDP | fit 3.97 s, 8 LOC incl. plot | fit 0.36 s, 9 LOC incl. plot | gamfit REML gives near-linear s(year) (edf 1.09, 7 unique years); pyGAM wiggles. Factor PD impossible in gamfit. Built-in `model.plot(x="age")` draws a zigzag. |
| Default credit LogisticGAM (10k rows) | gridsearch 204.5 s, acc 0.9739 | did not finish: >14 min, then killed at 25 min; 40+ inner-loop failures (DOC-17) | String Yes/No works only with explicit `family="binomial"`. |
| mcycle wiggly | gridsearch 0.23 s, PI coverage 0.955 | 0.44 s, edf 10.0, obs-interval coverage 0.955; location-scale `noise_formula="s(times)"` edf 18, coverage 0.97, heteroscedastic bands | Showcase: pyGAM cannot model the variance. |
| Poisson (faithful histogram) | 0.48 s | 1.67 s; `family` auto picks Poisson/log on integer counts | fine |
| Monotone constraint (300 rows) | 0.04 s | 55.36 s | severe latency on the documented shape-constraint path |
| te() interaction (1500 rows, sin(a)cos(b)+0.5ab) | 74.64 s, rmse vs truth 0.0312 | 7.25 s, rmse vs truth 0.0359 | gamfit PD for te needs a hand-built meshgrid (t4b error, t4c_te_grid.png works); pyGAM `generate_X_grid(meshgrid=True)` does it for you. |

## Findings

(Ids DOC-n. kind in {gap, bug, pyGAM-slop-to-avoid, already-better}.)

DOC-1 bug / blocker. `gamfit.fit(pandas_df, ...)` raises `ImportError: Import pyarrow failed` on
pandas 3 without pyarrow. Evidence: t0_pandas_no_pyarrow.py; HEAD gamfit/_tables.py:108-123 routes any
pandas frame that has `__arrow_c_stream__` (all pandas 3 frames) through the Arrow C stream, which
pandas implements by importing pyarrow. pyproject.toml:45 core deps omit pyarrow; it is only in the
`pandas` extra (line 72). getting-started.md:32 says "gamfit runs without any extra"; README_PYPI says
pandas is "accepted without conversion". The docs CI (tests/test_documentation_examples.py) builds
pandas frames in an env that has pyarrow, so it never sees this. A pyGAM user's first
`gamfit.fit(df, ...)` fails. Fix: in _tables.py take the Arrow path only when pyarrow is importable
(`_try_import("pyarrow")`), otherwise use the existing per-column path; add a no-pyarrow CI job.
Files: gamfit/_tables.py, CI workflow. Size S.

DOC-2 bug / high. `model.plot(data, x=...)` does not draw partial effects; README.md:326 says it
"draws partial effects and residuals". HEAD gamfit/_diagnose_plot.py:73-160: kind="prediction"
predicts the full model on each training row and draws it against x sorted, so any multi-covariate
model gives a zigzag (t1_wage_gamfit_builtin_plot.png, wage ~ s(year)+s(age)+education). Fix: make
the README claim true by adding the term plot (DOC-3) and either restricting kind="prediction" to
one-covariate models or having it call the Rust partial dependence for the named column's term.
Files: README.md, README_PYPI.md, gamfit/_diagnose_plot.py. Size S.

DOC-3 gap / high. No per-term partial-effect plot helper (pyGAM's defining workflow). The
computation already exists in Rust (crates/gam-inference/src/partial_dependence.rs, exposed as
`Model.partial_dependence`, HEAD gamfit/_model.py:1154), but the result has no interval bounds, so
users write `mu +- 1.96*se` by hand (a magic constant in user code, and wrong on the response scale
for non-identity links). gamfit already ships SAE plot helpers (gamfit/_sae_viz.py) and
`plot_trace` (gamfit/_sampling.py:355) but nothing for smooth terms. Fix: (a) Rust PD returns
`lower`/`upper` for a requested `interval` level (and optionally response-scale transforms),
(b) `Model.plot_terms(terms=None, interval=0.95, rug=True, ax=None)` in Python that only draws what
Rust returns (1D line+band, 2D contour for te/ti, factor level dots). Files:
crates/gam-inference/src/partial_dependence.rs, crates/gam-pyffi (PD binding),
gamfit/_diagnose_plot.py, gamfit/_model.py. Size M.

DOC-4 gap / high. Partial dependence is undocumented. `partial_dependence` appears only in
docs/frontend-parity.md:77; absent from api-reference.md, predictions.md, getting-started.md,
cookbook.md and mkdocs nav. The wheel on PyPI (0.1.267) has a different signature
(`term, data, grid, n_points`) than HEAD (`term, grid, n_points`), so users following any future doc
on the released wheel will fail. Fix: an "Inspecting terms" page (partial effects, bands,
term_blocks, variance_share, smooth_significance, difference_smooth) linked from nav and from
getting-started; api-reference entry. Files: docs/partial-effects.md (new), docs/api-reference.md,
docs/predictions.md, mkdocs.yml. Size S.

DOC-5 gap / med. Partial dependence refuses factor terms. Wheel: "cannot infer a 1D sweep axis
from term 'education'". HEAD resolve_term (partial_dependence.rs:282-359) only looks in
linear_terms and smooth_terms and refuses FactorSumToZero/factor-by blocks, so a bare factor
(stored as a random-effect block) still errors. pyGAM `f(2)` partial dependence is a standard tour
panel. Fix: support categorical terms by returning one row per saved level (estimate, SE) from the
coefficient block; the plot helper draws points with bars. Files: partial_dependence.rs. Size S/M.

DOC-6 gap / med. `print(model.summary())` has no term table. HEAD gamfit/_summary.py:462-496 prints
formula, family, deviance, REML, total edf, coefficient count only; the per-term edf/ref_df/chi_sq/
p_value already exist in `summary.smooth_terms`. pyGAM's `summary()` prints a term table; mgcv
too. Fix: render the smooth-terms table (Rust `summary_*` renderer owns it, as it already does for
the HTML report) in `__str__`. Files: gamfit/_summary.py, crates gam-report renderer. Size S.

DOC-7 gap / low. `coefficients_frame()` rows have no names (name None), so users cannot map factor
level or parametric coefficients. Fix: carry column names from the design through to the summary.
Size S.

DOC-8 bug / med (doc wrong). docs/formulas.md:101,117 call a bare `+ g` / `factor(g)` a FIXED
factor. Measured (t1d.py): bare, `factor()` and `group()` give identical smoothing parameters and
the education block is kind "random_effect" (penalized); only unseen-level prediction differs (bare
and factor raise GamError, group predicts the population level). Either the docs or the behaviour is
wrong; pyGAM users expect `f()` = unpenalized dummy coding. Fix: decide semantics, then make the doc
state it (if penalized, say so and explain why REML-shrunk factor effects are the better default).
Files: docs/formulas.md. Size S (doc) or M (behaviour).

DOC-9 bug / low. docs/diagnostics.md markdown table is broken: a paragraph (lines 9-18) sits inside
the table started at line 5, so rows 19-23 render as stray text. Size S.

DOC-10 gap / high. No "migrating from pyGAM" guide. grep for pygam across docs/ finds only the
internal issue-1082-validation.md. Outline and mapping table below. Files: docs/migrating-from-pygam.md,
mkdocs.yml, README link. Size M.

DOC-11 gap / high. README first screen does not say why gamfit is better. README.md opens with a dense
scope list and a 3D Matern surface; SAE/marginal-slope/event-history paragraphs dominate; no mention
of pyGAM, mgcv, REML-vs-GCV, speed or uncertainty. The Usage example uses undefined `train`/`test`
and has no plot. Fix: first screen = one sentence pitch, a 10-line runnable example on a real
dataset with a partial-effect figure, 3 bullet "vs pyGAM" claims each linked to evidence (REML not
GCV/grid, posterior-mean predictions with calibrated intervals, location-scale), then the long
feature list. Files: README.md, README_PYPI.md. Size S.

DOC-12 gap / med. No published benchmark vs pyGAM. Internal numbers exist
(bench/gha_results/reference-quality/quality_results.tsv: e.g. line 428 lidar rmse 0.0706 vs pyGAM
0.0709; line 317 logistic 1D shape pearson 0.998; lines 410-411 Poisson 2D shape TIMEOUT >360 s)
but nothing in docs/. Fix: a generated docs/benchmarks.md from that TSV (accuracy, interval
coverage, wall time), including the losses (timeouts, monotone latency) honestly. Size M.

DOC-13 gap / med. No datasets or tour. Docs examples use 12-20 hand-typed rows or rely on names
injected by tests/test_documentation_examples.py (df, X, y, train, test_df), so a reader who
copy-pastes cookbook.md (which says "Runnable examples" at line 3) gets NameError. Fix: docs/tour.md
reproducing wage, Default, mcycle (with location-scale), Poisson, monotone, te(), each self-contained
(load via URL/`pd.read_csv` of Rdatasets or a tiny `gamfit.datasets` fetcher). Size M.

DOC-14 pyGAM-slop-to-avoid / med. docs/cookbook.md:344-358 and docs/sklearn.md:144-158 show
`GridSearchCV` over formulas. SPEC forbids grid search; this is exactly pyGAM's `gridsearch` habit
in a new form. Fix: replace with REML-based model comparison (`gamfit.compare_models`, AIC/LAML) or
remove. Size S.

DOC-15 gap / med. Log and warning noise on trivial fits with no documented off switch. A 200-row
`y ~ s(x)` fit prints "[OUTER] ARC cost-stall STUCK ..." lines (crates/gam-solve/src/rho_optimizer/
bridges.rs:2711 uses log::warn! for optimizer internals; DEFAULT_LOG_LEVEL=Warn in
crates/gam-solve/src/progress_log.rs) plus a GamInferenceWarning about auto knots on every default
fit. progress_log.rs refers to a "Python set_log_level shim" that is not exported. Fix: demote
optimizer-internal chatter to debug/trace; export `gamfit.set_log_level`; do not warn on the
default basis size. Size S.

DOC-16 pyGAM-slop-to-avoid / med. The auto-knot rule `clamp(unique/4, 4..8)` is a magic constant
announced in a warning on every default fit (SPEC: no magic knobs). pyGAM's `n_splines=20` is the
same kind of arbitrary default. Note for the terms auditor; docs should explain that basis size is an
upper bound and REML sets wiggliness, not the knot count. Size S (doc).

DOC-17 bug / high. Tour example 2 (ISLR Default, 10,000 rows,
`d ~ student + s(balance) + s(income)`, binomial 0/1) did not finish in gamfit on two attempts:
more than 14 min (first run, t2.err) and killed by `timeout 1500` (25 min wall) in the rerun
(t2b.err, last log line at 17 min 02 s). pyGAM's full gridsearch finished on the same machine and
load in 204.5 s, acc 0.9739. The gamfit log shows 40+ "P-IRLS INNER LOOP FAILED" lines: Hessian
not positive definite with minimum eigenvalue down to -3.1e171, condition number inf, "did not
converge within 300 iterations, last gradient norm 2.2e149", plus "BFGS Strong Wolfe failed at
iter 32". Caveat: load average 35-44 on 4 CPUs inflates every timing, but the ratio to pyGAM and
the divergent inner iterates are not a load artifact. A possible (unverified) cause is the quasi-separable tail of
balance (default rate near 0 below 1000), which drives the logit iterates to huge magnitudes.
Hand to the speed/robustness auditors for a quiet-machine repro before calling it a blocker.
Fix belongs in the P-IRLS step control in Rust, not in docs. Until it is fixed, the migration
guide cannot use this example.

DOC-18 friction / low. String Yes/No response to `gamfit.fit` raises InvalidConfigurationError
suggesting `family='binomial'`. Verified (t2b.out): with `family="binomial"` the Yes/No column
fits (Binomial Logit), so the message is accurate. Remaining friction is that family auto-detect
does not choose binomial for a two-level label column even though the error already knows it is
binary. Docs should show `family="binomial"` with string labels in the logistic example and state
which level is the event. GAMClassifier accepts labels.

DOC-19 gap / low. mkdocs.yml publishes internal issue-*.md / test-census pages from docs_dir
(unlisted but searchable). Move them out of docs_dir or add `exclude_docs`. Size S.

DOC-20 gap / low. cookbook.md lines 380-391 are three one-line sections appended after
"Example index". Size S.

DOC-21 gap / low. `partial_dependence` on a 2D term (te, ti, thin-plate) with no `grid` raises
"cannot infer a 1D sweep axis ... Multi-dimensional smooths always require an explicit grid"
(wheel, t4b.out). HEAD is the same: partial_dependence.rs:191-195 has a default grid only for one
axis. Users build a meshgrid and reshape by hand; pyGAM's `generate_X_grid(term, meshgrid=True)`
does it for them. Fix: TrainingRange default builds an `n_points`-per-axis product grid over the
training box (a hull mask is optional) and returns the axis vectors so Python can reshape without
guessing. Files: partial_dependence.rs. Size S.

DOC-22 bug / high (hand to speed auditor). The documented shape-constraint path is very slow:
`y ~ s(x, shape=monotone_increasing)` on 300 rows took 55.36 s, versus 0.04 s for pyGAM
`s(0, constraints='monotonic_inc')` (t4.out, same load). Monotonicity of the result was not checked,
because the script was cut off before that line. A constraints tour page cannot ship with that
latency. Size: engine work, not docs.

## Already better than pyGAM (use these in the docs)

- AB-1 REML smoothing on wage: s(year) edf 1.09 (near linear, 7 unique years) versus pyGAM's
  default-lambda wiggle; no gridsearch needed; fit 0.36 s vs 3.97 s.
- AB-2 mcycle location-scale: `noise_formula="s(times)"` gives heteroscedastic observation bands
  (coverage 0.97) that pyGAM cannot express; default fit matches pyGAM coverage 0.955 without
  gridsearch. Figure t3_mcycle.png is ready-made README material.
- AB-3 Posterior-mean predictions with `posterior_mean_lower/upper` and `observation_lower/upper`
  columns from one `predict(interval=0.95, observation_interval=True)` call; pyGAM needs two methods.
- AB-4 `family` auto-detects Poisson/log on integer counts.
- AB-5 `model.sample` (posterior draws, 0.15 s on mcycle) vs pyGAM `sample`, which bootstraps and refits.
- AB-7 te() interaction: 7.25 s vs pyGAM 74.64 s on 1500 rows (both under load), comparable
  error vs truth (0.036 vs 0.031), and PD comes with smoothing-corrected SE surfaces
  (t4c_te_grid.png); pyGAM's te intervals use the conditional covariance only.
- AB-6 Executable-docs CI already exists (tests/test_documentation_examples.py) - but it injects
  globals, hiding NameErrors from readers (DOC-13) and runs with pyarrow (DOC-1).

## Migration guide outline (docs/migrating-from-pygam.md)

1. Why switch (4 bullets, each linked to evidence: REML/LAML instead of GCV/UBRE + grid search;
   posterior-mean predictions and calibrated intervals; location-scale and more families;
   Rust speed with benchmark link).
2. Mental model: formulas on named columns instead of column indices; one `gamfit.fit` instead of
   one class per family; smoothing chosen by REML, never searched.
3. Side-by-side mapping table.
4. Worked example: wage tour in both libraries, same figure.
5. Things pyGAM did that gamfit deliberately does not (gridsearch, GCV/UBRE, `lam=` tuning,
   `n_splines` as a smoothness knob) and why.
6. Gotchas: pandas without pyarrow (until DOC-1 lands), factor semantics, log level.

| pyGAM | gamfit |
|---|---|
| `LinearGAM(s(0)+s(1)+f(2)).fit(X, y)` | `gamfit.fit(df, "y ~ s(a) + s(b) + factor(c)")` |
| `LogisticGAM`, `PoissonGAM`, `GammaGAM`, `InvGaussGAM`, `ExpectileGAM` | `gamfit.fit(..., family="binomial"/"poisson"/"gamma"/...)`, auto-detect by default |
| `GAM(distribution=, link=)` | `family=`, `link=` |
| `.gridsearch(X, y, lam=...)` | not needed: smoothing parameters are REML/LAML optimized |
| `s(0, n_splines=20, lam=0.6)` | `s(a, k=20)`; lambda is estimated, not set |
| `s(0, spline_order=3)` | `s(a, degree=3)` (formulas.md:131,153) |
| `s(0, constraints='monotonic_inc')` | `s(a, shape=monotone_increasing)` |
| `s(0, basis='cp')` | `cyclic(a, period=...)` (aliases cc/cp, formulas.md:189) |
| `te(0, 1)` | `te(a, b)` |
| `l(0)` | `a` or `linear(a)`; `linear(a, min=0)` for sign constraints |
| `s(0, by=2)` | `s(a, by=c)` (factor or numeric by, formulas.md:102-103) |
| `gam.predict(X)` | `model.predict(new_df)` (posterior mean) |
| `gam.predict_mu(X)` | `model.predict(new_df)` |
| `gam.confidence_intervals(X, width=.95)` | `model.predict(new_df, interval=0.95)` -> `posterior_mean_lower/upper` |
| `gam.prediction_intervals(X, width=.95)` | `model.predict(new_df, interval=0.95, observation_interval=True)` -> `observation_lower/upper` |
| `gam.generate_X_grid(term=i)` + `gam.partial_dependence(term=i, X=XX, width=.95)` | `model.partial_dependence("s(a)")` (grid built in Rust); bands: see DOC-3 |
| `gam.summary()` | `model.summary()`; `.smooth_terms_frame()` for the table |
| `gam.statistics_['edof']` | `model.summary().edf_total` |
| `gam.statistics_['p_values']` | `summary().smooth_terms_frame()["p_value"]`, `model.smooth_significance(df)` |
| `gam.accuracy(X, y)`, `gam.score` | `model.diagnose(df).metrics`; sklearn `GAMClassifier.score` (AUC) |
| `gam.sample(X, y, n_draws=)` | `model.sample(df, samples=, seed=)` (posterior draws via NUTS per docstring, no bootstrap refit) |
| `gam.deviance_residuals(X, y)` | `model.diagnose(df)` residuals |
| sklearn usage of `LinearGAM` | `gamfit.sklearn.GAMRegressor` / `GAMClassifier` |
| `pygam.datasets.wage()` | no bundled data; `pd.read_csv(<Rdatasets URL>)` |
| GCV / UBRE objective | never; REML/LAML only (slop to avoid) |


## Proposed deliverables (SPEC-compliant: plotting in Python, all numbers from Rust)

| # | Deliverable | Files | Size |
|---|---|---|---|
| D1 | pyarrow-optional pandas input + no-pyarrow CI job | gamfit/_tables.py, .github/workflows | S |
| D2 | PD returns interval bounds at a level (and response scale) | crates/gam-inference/src/partial_dependence.rs, crates/gam-pyffi | S |
| D3 | PD for factor terms (per-level estimate+SE) | partial_dependence.rs | S/M |
| D4 | `Model.plot_terms()` (1D band+rug, 2D contour, factor points), draws Rust output only | gamfit/_diagnose_plot.py, gamfit/_model.py | M |
| D5 | Fix `plot(kind="prediction")` zigzag and the README claim | gamfit/_diagnose_plot.py, README*.md | S |
| D6 | Term table in `print(summary)` (Rust renderer) | gamfit/_summary.py, gam-report | S |
| D7 | Coefficient names in coefficients_frame | summary plumbing | S |
| D8 | docs/partial-effects.md + api-reference entries + nav | docs/, mkdocs.yml | S |
| D9 | docs/migrating-from-pygam.md (outline above) | docs/, mkdocs.yml | M |
| D10 | docs/tour.md (self-contained real-data tour, figures) | docs/, docs/img | M/L |
| D11 | docs/benchmarks.md generated from bench TSV | bench/, docs/ | M |
| D12 | README first-screen rewrite | README.md, README_PYPI.md | S |
| D13 | Replace GridSearchCV recipes with REML comparison | docs/cookbook.md, docs/sklearn.md | S |
| D14 | Fix formulas.md factor wording (or behaviour) | docs/formulas.md | S |
| D15 | Fix diagnostics.md table; cookbook tail; exclude internal docs | docs/diagnostics.md, docs/cookbook.md, mkdocs.yml | S |
| D16 | `gamfit.set_log_level`, demote optimizer chatter, drop default-knot warning | gam-solve progress_log.rs, bridges.rs, gamfit/__init__.py | S |
| D17 | Make doc snippets self-contained (stop injecting df/X/y) | tests/test_documentation_examples.py, docs | M |
| D18 | Default product grid for multi-axis PD terms | partial_dependence.rs | S |
