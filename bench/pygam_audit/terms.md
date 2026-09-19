# Audit axis: term types, bases, shape constraints (gamfit 0.1.267 vs pyGAM 0.12.0)

All scripts live in `scratchpad/audit/terms/`. They were run with the scratch venv from that directory; nothing under /home/user/gam was modified and no cargo build was run.
The machine load average was about 37 on 4 CPUs throughout, so absolute timings are inflated and only the relative timings mean anything.

| script | purpose |
|---|---|
| `t01_monotone.py` | 5 shape truths × 5 seeds, n=150. Checks shape on a dense grid in-sample `[0,1]` and extrapolated `[-0.5,1.5]`. |
| `t02_pygam_soft_violation.py` | pyGAM soft-penalty violation vs response scale and n. |
| `t03_mono_accuracy.py` / `t03b.py` (`t03b.rows`) | RMSE and failure rate: pyGAM vs gamfit unconstrained / shape / shape k=20. |
| `t04_feature_probe.py` | about 65 DSL probes: accepted or rejected. |
| `t05_pygam_combos.py` | pyGAM te-margin monotone, by= monotone, multiple constraints, cp+monotone, convex+concave. |
| `t07_diag.py`, `t08_ident.py` | λ and coefficient diagnosis of monotone fits; intercept aliasing. |
| `t09_interval_hang.py`, `t10_interval_scaling.py` | Cost of `predict(interval=)` on a shape-constrained fit. |

## Feature map

| pyGAM | gamfit equivalent | status |
|---|---|---|
| `s(i)` | `s(x)` (B-spline, exact ∫f''² penalty + null-space double penalty, REML) | better |
| `l(i)` | `x` / `linear(x)` | ok |
| `f(i)` | `factor(g)` / bare string column (fixed effect); `s(g, bs=re)` / `group(g)` (random) | better |
| `te(i,j)` | `te(x,z)` with per-margin `k`, `degree`, `penalty_order`, `bs=[ps,cr]`, `periods=[None,24]`; plus `ti()` and 3-D te | better |
| `intercept`, `fit_intercept=False` | intercept always on; `-1` / `0+` rejected (`formula_dsl.rs:2663` "not supported yet") | **gap** |
| term addition `+` | `+` | ok |
| `n_splines` | `k=` / `knots=` (default clamp(unique/4,4,8) interior knots, documented in notes) | ok |
| `spline_order` | `degree=` 0..5 accepted | ok (but see B-parse) |
| `lam` per term | none. λ always from REML (SPEC) | by design, not a gap |
| `penalties='derivative'/'l2'/None` | `penalty_order=`, `double_penalty=` (null-space ridge). `penalty=none`, `fx=true` rejected | ok by SPEC. Multi-order silently ignored (bug) |
| `constraints='monotonic_inc/dec'`, `'convex'`, `'concave'` | `shape=monotone_increasing/...`, `constraints={"s(x)": kind}` kwarg | hard cone, better in kind. **Accuracy bug**, see A1 |
| constraints list `[inc, concave]` | rejected (`shape=[..]`, `'a,b'`) | **gap** |
| constraints + `by=` | rejected for `by=g` and `by=z` | **gap** |
| te margin constraint | rejected (`te(..., shape=...)` scalar or list) | **gap** |
| `basis='cp'` | `cyclic(t, period=)`, `s(t, period=24)`, `cp(...)` | ok |
| `basis='cp'` + monotone | refused (`shape_constraints.rs:37-55`) | better. pyGAM accepts it and violates by 6.3e-2 |
| `by=` numeric | `s(x, by=z)`, `te(x,z, by=w)` | ok |
| `by=` factor | `s(x, by=g)`, `te(x,z, by=g)` (pyGAM cannot) | better |
| `edge_knots` | DSL `range=/domain=/boundary_knots=` rejected. Python `smooths={"x": gamfit.BSpline(knots=...)}` descriptor exists in code (`gamfit/_api.py:944-970`) | DSL gap (parity) |
| `dtype='categorical'` | `factor(gi)` | equivalent |

## Findings

### A1 — bug / HIGH — Adding a monotone constraint makes gamfit less accurate than its own unconstrained fit and than pyGAM

`t03b` setup: n=200, noise sd 0.3, x ~ U(0,1), 5 seeds (rng 100+seed). The table gives RMSE against the truth on a 1001-point grid, with `/fN` = number of fit failures.

| truth | pyGAM (gridsearch, soft) | gamfit `s(x)` | gamfit `s(x, shape=…)` | gamfit `s(x, k=20, shape=…)` |
|---|---|---|---|---|
| inc_step (sigmoid, slope 30) | 0.0456 /f0 | 0.0547 /f0 | 0.0764 /f1 | 0.0886 /f1 |
| inc_flat (hinge at 0.6) | 0.0370 /f0 | 0.0364 /f0 | 0.0613 /f0 | 0.0740 /f2 |
| dec_exp (exp(-4x)) | 0.0411 /f0 | 0.0339 /f0 | 0.0386 /f2 | 0.0437 /f1 |

- On inc_* the constrained fit is worse than gamfit's own unconstrained fit in 9 of 9 successful paired fits, often by 1.6–1.8×.
- On dec_exp it is worse in 3 of 3.
- `t01` (n=150, 5 seeds) showed the same pattern: inc_step 0.082–0.103 vs pyGAM 0.057–0.071.
- A shape constraint should never cost accuracy when the truth satisfies it.

Diagnosis (`t07_diag.py`, `t08_ident.py`, seed rng 102, inc_step):

- **λ falls when the constraint is added.**
  - Unconstrained sp = {52.5, 0.036}; constrained sp = {27.7, 0.0006}.
  - Seed 103: 56.2 → 8.4.
  - The constrained fit is *wigglier*, although the cone removes variance and should permit less smoothing, not more.
- **The reported point drifts in flat regions.**
  - Fitted range is −0.20…1.13 against truth 0…1.
  - The transformed coefficients (level + increments γ ≥ 0) are `[0.065, 0.085, 0.047, 0.038, 0.195, 0.568, 0.106, 0.042, 0.048, 0.063, 0.055]`.
  - Every flat-region increment sits at about 0.04–0.08, roughly 0.8σ, which is the half-normal mean. None sits on its bound.
  - The reported point is the truncated-Laplace posterior mean (`gam-solve/src/constrained_posterior.rs:1-76`). With m flat increments each shifted up by about 0.8σ, the curve drifts by about 0.8·m·σ across a flat region.
  - k=20 is worse than the default k in every case, which is consistent with drift growing with the number of constrained increments.
- **The 95% credible band excludes the truth at both ends** (`t09.out`):
  - x=0: truth ≈ 0.0025, band [−0.436, −0.064].
  - x=1: truth ≈ 1.0, band [1.010, 1.300].
- The LAML already carries the posterior orthant normalizer (`constrained_posterior/cone_normalizer.rs`). That file never mentions the prior's truncation mass P_prior(C | ρ). It should enter the evidence of a truncated prior, `∫_C L·π / P_π(C)`, and with two penalties (curvature + null-space ridge) P_π(C) depends on ρ. This is a hypothesis for the λ drop, not confirmed.

Proposed principled fix (L). This stays inside the SPEC: posterior mean, REML/LAML, no knobs.

1. Make the outer criterion the evidence of the model actually reported: the truncated-prior marginal likelihood, `log ∫_C p(y|β)π(β|ρ)dβ − log P_π(C|ρ)`.
   - The prior orthant mass is a Gaussian orthant probability of the *prior* precision restricted to the constrained coordinates.
   - The existing EP orthant code in `cone_normalizer.rs` computes it; the improper null-space directions are unconstrained and drop out.
2. Add a regression benchmark as a test (not an XFAIL) on the inc_step / inc_flat / dec_exp truths above. Assertions:
   - constrained RMSE ≤ unconstrained RMSE on average;
   - constrained λ ≥ unconstrained λ in flat-truth cases;
   - band coverage at the endpoints.
3. If a correct evidence still leaves the drift (it is the Bayes mean of a prior with zero mass on faces), the principled alternative is a prior with mass on faces:
   - an exponential or half-normal prior on the increments γ_j, equivalent to adding the linear term `κ·1ᵀγ` with κ REML-estimated;
   - so that the posterior mean of a flat increment is O(σ²κ) rather than 0.8σ.
   - Do not "fix" it by switching the point to the mode, which the SPEC forbids.

Files:

- `crates/gam-solve/src/constrained_posterior/cone_normalizer.rs`
- `crates/gam-solve/src/reml/reml_outer_engine/objective.rs` (`cone_normalizer_outer_hessian` ~2160)
- `crates/gam-solve/src/constrained_posterior.rs`
- a new benchmark test

### A2 — bug / HIGH — Shape-constrained s() is exactly aliased with the intercept

`term_specs.rs:8222-8226` forces `identifiability = None` for every shape term ("anchored by construction"). The B-spline basis is a partition of unity, so the term's level coordinate duplicates the intercept.

Evidence (`t08_ident.py`, same data):

- `y ~ s(x)`: intercept sd = 0.021, corr(b0,b1) = 0.0000.
- `y ~ s(x, shape=monotone_increasing)`: intercept sd = **28.35**, corr(b0,b1) = **−1.0000**, and all 12 raw basis columns are retained.
- `s(x, double_penalty=false, shape=monotone_increasing)` fails with "Pre-fit rank deficiency … rank 1 < 2 unpenalized columns … columns [0, 1]", while `s(x, double_penalty=false)` fits (`t07`).
- The model is identified only by the null-space ridge, which shrinks the level of f against the intercept.
- The summary intercept SE is meaningless.

Fix (S): in the shape chart f = c·1 + Σ γ_j C_j with the cone on γ only, drop the level coordinate c when the model has an intercept. Then centre the remaining columns (C_j − mean C_j). Centring is an affine shift of c, so it cannot break the cone. The comment "sum-to-zero side constraints conflict with monotonic/convex cones" is incorrect for this parameterization.

Files: `crates/gam-terms/src/smooth/term_specs.rs:8222`, `crates/gam-terms/src/smooth/shape_constraints.rs`.

### A3 — bug / HIGH — Shape-constrained fits fail to converge (hard error) on ordinary data

- `t03b`: 3 of 15 default-k shape fits and 5 of 15 k=20 fits raise.
- `t01`: 2 of 25 fits raise (inc_step seed 3, dec_exp seed 1; `rng=np.random.default_rng(seed); x=U(0,1,150); y=f(x)+N(0,.3)`).
- The error is "Outer smoothing-parameter optimization did not certify a stationary optimum … termination=iteration_budget(200 iterations)".
- pyGAM never fails on the same data.
- The SPEC rightly forbids returning an unconverged fit, but these are easy 1-D problems.

Fix (M-L): the outer ρ search must handle the (now continuous, per `cone_normalizer.rs`) criterion near face switches. Two changes:

- a safeguarded 1-D/2-D trust region with a face-aware line search that recomputes the active set per trial ρ;
- certifying stationarity of the continuous criterion rather than hitting a 200-iteration budget.

Fixing A1/A2 (aliasing plus the ridge-only identification) is likely to remove part of the ill-conditioning.

Files: `crates/gam-solve/src/rho_optimizer/*`, `crates/gam-solve/src/reml/reml_outer_engine/*`.

### A4 — bug / HIGH — Constrained KKT gate is not scale-invariant

Repro: n=20000, `y = 1e4*(x - 0.6*exp(-((x-.5)/.05)**2) + N(0,.05))`, `s(x, k=30, shape=monotone_increasing)` fails with:

> "KKT residuals exceed tolerance: primal=5.4e-13, comp=1.290e-6 … stat_rel=2.8e-14"

The identical data at unit scale fits.

Cause: `crates/gam-solve/src/reml/gradient_hessian.rs:4420-4425`. Stationarity was made relative (#989), but primal, dual and complementarity are compared with absolute `KKT_TOL_* = 1e-7` (`reml/state_caches.rs:33-37`). Multipliers scale with the gradient, O(n·scale²).

Fix (S): gate `complementarity / max(1, ‖grad‖∞·max(1, ‖slack‖∞))`, and primal relative to `‖A‖·‖β‖`, mirroring the stationarity treatment.

### A5 — bug / HIGH (perf) — `predict(interval=…)` on a shape-constrained fit costs seconds per prediction row

`t09`, 200-row monotone fit:

- fit 8.0 s;
- `predict` without interval, 5 points: 0.0 s;
- `predict(interval=0.95)`, 5 points: **32.6 s**;
- 201 or 501 points did not finish in 900 s (`t06`, `t08`, two independent hangs).

`t10`: the cost is linear in rows, about 6.5 s per row under load.

| model | 1 row | 20 rows |
|---|---|---|
| `s(x)` | 0.0 s | 0.0 s |
| `s(x, shape=monotone_increasing)` | 10.5 s | 129 s |

Likely cause: a per-row truncated-moment / EP solve in the constrained prediction path (`crates/gam-predict/src/lib.rs:1777-1830`, `constrained_ambient_covariance` and the truncated-law code). The truncated posterior moments (E_π β, Cov_π β) are row-independent: compute them once per fit and apply `x_rowᵀ Cov_π x_row` per row.

Fix (S-M). Files: `crates/gam-predict/src/lib.rs`, `crates/gam-solve/src/constrained_posterior.rs`.

### G1 — gap / HIGH — No shape constraint on a te() margin

- pyGAM: `te(0,1, constraints=['monotonic_inc', None])` is monotone in x on 41 z-slices, with worst violation 0.0, including extrapolation to x∈[−0.5,1.5], z∈[−0.2,1.2] (`t05`).
- gamfit rejects `te(x,z, shape=monotone_increasing)` and `shape=[monotone_increasing, none]` (`t04`).

Design (M-L):

- For te of open B-spline margins, f = Σ β_ij B_i(x) C_j(z) with C_j ≥ 0.
- The rows (D_x ⊗ I_z)β ≥ 0 certify ∂f/∂x ≥ 0 for every z, exactly and on the continuum. Convexity uses the span-scaled D2 ⊗ I in the same way.
- Build the rows with `bspline_shape_linear_constraints` per margin and Kronecker them. Use the general linear-inequality path, since box reparameterization does not apply to Kronecker cones.
- The te identifiability transform must stay a level-only shift, so that it preserves the chart (as in A2).

Files: `crates/gam-terms/src/term_builder.rs` (tensor path ~3817), `smooth/shape_constraints.rs`, `smooth/term_specs.rs` (the tensor spec gets a per-margin `shape` list).

### G2 — gap / MED — Several constraints on one term

- pyGAM `constraints=['monotonic_inc','concave']` holds: inc viol 0, concave viol 5.8e-12 (`t05`).
- gamfit `ShapeConstraint` is a single enum (`term_specs.rs:70-76`); `shape=[..]` and `'a,b'` are rejected (`t04`).

Design (M):

- `shape` becomes a set.
- Rows = stacked exact rows from `bspline_shape_linear_constraints` for each order, merged with the existing `merge_linear_constraints_global` (`shape_constraints.rs:195`).
- Contradictions reduce the cone explicitly: inc+dec gives a constant, convex+concave gives an affine f. Reject or explain them; do not silently return a degenerate fit.
- Only the single-order case keeps the box reparameterization.

Files: `term_specs.rs`, `shape_constraints.rs`, `term_builder.rs:~575`.

### G3 — gap / MED — Shape constraint with `by=`

- pyGAM `s(0, by=1, constraints='monotonic_inc')` holds, worst decrease 0 (`t05`).
- gamfit rejects `s(x, by=g, shape=…)` and `s(x, by=z, shape=…)` (`t04`; `term_specs.rs:7998`, `:8601`).

Design (S-M):

- Factor by: block-diagonal copies of the same exact rows, one per level block.
- Numeric by: the cone is on f's coefficients, and the product z·f(x) is documented as monotone in x only where z ≥ 0.

Files: `smooth/term_specs.rs` (by expansion and factor-smooth paths).

### G4 — gap / MED — No intercept removal

- `y ~ s(x) - 1`, `0 + s(x)` and `-1 + s(x)` are rejected: "formula terms '0'/'-1' (intercept removal) are not supported yet" (`crates/gam-terms/src/inference/formula_dsl.rs:2663`).
- pyGAM has `fit_intercept=False`.

Fix (S-M): support `0+`/`-1` in the DSL. When no intercept is present, the first smooth keeps its level coordinate (`identifiability=none`), with the same logic as A2 in reverse.

Files: `formula_dsl.rs`, `term_builder.rs`.

### G5 — gap / LOW — No edge knots / domain in the DSL

- `s(x, range=[-1,2])`, `domain=` and `boundary_knots=` are rejected (`t04`).
- The Python `gamfit.BSpline(knots=...)` descriptor via `fit(..., smooths=...)` exists in code (`gamfit/_api.py:944-970`), so DSL and Python disagree (parity).
- Needed for extrapolation control and for matching pyGAM `edge_knots`.

Fix (S): a `boundary=[a,b]` option for B-spline s() and te margins, validated to contain the data range.

Files: `term_builder.rs` (knot construction), `formula_dsl.rs`.

### B-parse — bug / MED — Invalid `degree` / `penalty_order` values are silently ignored

- `s(x, degree=cubic)`, `degree=3.0`, `penalty_order=two`, `penalty_order=-1`, `penalty_order=[1,2]` and `penalty_order=all` all fit.
- Each is identical to plain `s(x)`, with sp {0: 225.416, 1: 2.185}.
- `k=twelve` and `knots=abc` are correctly rejected.

Cause: the lenient `option_usize` (`formula_dsl.rs:1847`, `.parse().ok()`) is used at `term_builder.rs:2492, 2672, 2766, 2810, 2981`. A strict `option_usize_strict` already exists at `formula_dsl.rs:1902`.

Fix (S): switch to the strict variant. Error on penalty_order > degree instead of the silent `.min(effective_degree)` at `term_builder.rs:2981`: degree=0 silently collapses to penalty_order 0/1, and the double penalty disappears (a single sp 44.9).

### B-doc — bug / LOW — Shape docs are wrong or missing

- The `gamfit/smooth.py:108` docstring says the constraint matrix is "generated from the basis on a dense 1D grid". The Rust code uses exact knot-based rows with no grid (`shape_constraints.rs:1-8`).
- `smooth.py:117` claims support for "thin-plate / Duchon with a single feature axis". `shape_supports_basis` accepts only open B-splines, and `duchon()`, `tps()` and `matern()` with shape are rejected (`t04`).
- `docs/formulas.md` never documents `shape=`.

Fix: S (docs only).

### B-noise — bug / LOW — stderr spam from constrained fits

Constrained fits print many `[GAM COST] P-IRLS INNER LOOP FAILED … minimum eigenvalue: -1.2e284` and `[INDEF-HESS]` dumps even when they succeed.

Fix: route these to the logging facade at debug level. S.

### Already better (keep)

- **Hard certified cone.**
  - gamfit monotone fits had 0 violations on a dense grid in-sample and extrapolated to [−0.5,1.5]; convex/concave ≤ 1e-14 (roundoff) (`t01`).
  - The rows are exact and knot-aware, with no grid.
- **Refuses meaningless combinations.**
  - periodic + monotone is refused. pyGAM accepts `basis='cp', constraints='monotonic_inc'` and returns a curve that decreases by 6.3e-2 and wraps f(0)=f(1)=0.870 (`t05`).
  - natural-cubic + shape, and boundary-condition + shape, are also refused, for correct reasons (`shape_constraints.rs:42-55`).
- **Penalty on the function:** the exact ∫(f^(m))² with a REML null-space penalty, vs pyGAM's coefficient differences (`docs/formulas.md:139-148`).
- **Richer term types:** factor `by=`, `te(by=)`, `ti()`, per-margin bs/degree/penalty_order/periods, and 3-D te.
- **REML** instead of GCV gridsearch; no `lam` knob.

### pyGAM slop to avoid (do not copy)

1. **Soft constraints.** A 1e9 quadratic penalty on masked coefficient differences, with the mask re-chosen inside PIRLS (`pygam/pygam.py:187-189`, `penalties.py:76-210`).
   - Violations are small in practice: about 1e-11 relative, independent of scale (`t02`).
   - But nothing is certified, and the 1e9 block enters the EDoF/GCV.
2. **Forced ridge on every constrained term.** `constraint_l2 = 1e-3·I` is added to every constrained term and escalated to 1e-1 on Cholesky failure (`pygam.py:520-535`, `terms.py:351-395`). It is an unestimated magic ridge that biases the fit.
3. **Convexity on raw coefficient second differences**, which ignores knot spacing. It is only a correct cone for uniform knots.
4. **Penalties on coefficients** (`penalties='derivative'` = difference matrix, `'l2'`) rather than on the function; per-term `lam`; GCV `gridsearch`.
5. **Accepting contradictory or meaningless constraint combinations** (cp + monotone; convex + concave) without comment.

### Not gaps (deliberately excluded)

- `lam=` / `sp=` / `fx=true` / `penalty=none`: SPEC rules forbid them (REML only, no knobs). REML can already drive λ→0.
- `dtype='categorical'` = `factor()`.
- pandas input needing pyarrow (`gamfit/_tables.py:110`) is off-axis and noted for the API auditor.
