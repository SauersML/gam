# Speed, memory, scaling audit: gamfit vs pyGAM

Axis: wall/CPU fit time, predict time, peak RSS, import / time-to-first-fit, scaling in n and p.
All proposed fixes are held to SPEC.md: REML/LAML only, no grid search, no derivative-free search,
no time budgets, no convergence corner-cutting, no magic knobs, the logic lives in Rust, and a fit
only comes from a converged optimization.

## 1. Method and environment

- **Machine.** 4 CPUs (`nproc`=4), 15 GB RAM with about 10 GB available, no swap. The box is
  shared with about 10 other agents: 1-minute load average ranged 33-48 (median 40) over every
  record. Each benchmark process got roughly 0.2 of a core.
- **Primary metric.** Because of that load, the primary speed metric is **CPU time** (`time.process_time`)
  of the fit and predict calls. Wall time is recorded but is inflated about 4-6x by the load.
- **Threads.** Every run is single-threaded (`RAYON_NUM_THREADS=OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=1`),
  so CPU time is comparable across libraries. pyGAM's OpenBLAS otherwise oversubscribes the box.
- **Fresh process.** Every measurement runs in a fresh subprocess with `cwd=/tmp`, so the installed
  wheel is used and not the repo source tree. RSS is taken two ways: the worker's `ru_maxrss`, and
  the driver polling psutil every 50 ms across the process tree.
- **Repetitions.** 3 reps (seeds 0, 1, 2) for n≤1e5, reported as median [min-max]. Where a table
  says 1 rep, the reason is given.
- **Versions.** gamfit wheel 0.1.267, pygam 0.12.0, numpy 2.4.6, pandas 3.0.6, Python 3.11.
  The repo is at HEAD d10950d (pyproject 0.1.268, Cargo 0.3.157).
  - **Caveat:** the wheel is one version behind the repo.
  - Measurements come from the wheel. Code evidence is cited at repo `file:line`.
  - Where the wheel and repo differ (for example `minv_xt` and `interval="full_conformal"` payload
    fields), the finding notes whether the repo still has the behavior.
- **Designs.**
  - `p1`, `p5`, `p20`: additive `s(x_j)` terms, truth `Σ sin(2πx_j + j)/√p`.
  - `te`: `te(x0, x1)` with truth `sin(2πx0)·cos(2πx1)`.
  - gamfit's default `s()` gives 11 coefficients and 2 penalties per smooth, so p5 has 56 coefficients
    with k=10 smoothing parameters, and p20 has 221 coefficients with k=40.
- **Library variants.**
  - `gamfit` runs at its defaults.
  - `gamfit_k20` sets `k=20` per term, roughly matching pyGAM's 20 splines.
  - `pygam` is `GAM().fit` with the fixed default lam=0.6.
  - `pygam_gs` is `GAM().gridsearch()`, an 11-point GCV/UBRE grid.
- **Accuracy check.** `rmse_mu` is the RMSE of the predicted mean against the true mean on 1e5
  held-out rows (or n rows), recorded only as a sanity check that fits are comparable.

## 2. Harness (permanent benchmark)

All harness files are in `scratchpad/audit/speed/`.

| file | purpose |
|---|---|
| `worker.py LIB FAMILY N DESIGN SEED` | One measurement: import, fit1 (cold), fit2 (warm, disabled with `BENCH_WARM=0`), pred1, pred2, ru_maxrss, rmse_mu, edf/ncoef/outer_iter or lam. Prints `RESULT {json}`. |
| `driver.py OUT.jsonl PLAN [--reps 3 --timeout S --memcap-mb M --taskset C --env K=V]` | Runs each rep in a fresh subprocess. Polls RSS and threads, records nproc and load average at start and end, and a status of ok/timeout/memcap/error. After a timeout or memcap it skips larger n for that config. Named plans: `n1e3`, `n1e4`, `n1e5`, `n1e6`, `n1e4_core`, `n1e5_core`, `n1e6_core`, `gaussian_all`, `glm_all`; a literal list of tuples also works. |
| `analyze.py FILE.jsonl...` | Per-config median [min-max] table, plus a gamfit/pyGAM ratio table with LOSS markers. |
| `trace_fit.py FAMILY N DESIGN [K]` | One fit with `gamfit._rust.set_log_level("info")`. stderr is the timestamped solver trace used for phase attribution. |
| `pred_overhead.py FAMILY N` | Predict CPU per call on 1000 rows, plus cProfile of the Python layer. |
| `model_bytes.py` | Size of the saved model payload against n. |
| `prof_fit.py`, `agg_py.py`, `cold_profile.py`, `cprof_warm.py` | py-spy and cProfile helpers. The `.so` is stripped, so native frames are unsymbolized and Rust phases are attributed from the info/debug trace instead. |

Rerun, for example:

```
cd /tmp && P=.../bvenv/bin/python; D=.../audit/speed
$P $D/driver.py $D/n1e4.jsonl n1e4_core --reps 3 --timeout 900 --memcap-mb 5000 \
   --env RAYON_NUM_THREADS=1 --env OPENBLAS_NUM_THREADS=1 --env OMP_NUM_THREADS=1 --env BENCH_WARM=0
$P $D/analyze.py $D/n1e3.jsonl $D/n1e4.jsonl $D/n1e5.jsonl $D/n1e6.jsonl
```

Suggested CI gate:
- Track the CPU ratio against `pygam_gs` for Gaussian, binomial and Poisson × p1, p5, te at n=1e4.
  gridsearch is the fair comparator, since it is the only pyGAM mode that chooses smoothness.
- Track the RSS ratio against `pygam` at n=1e5.
- Treat any p20 timeout as a hard failure.
- Set `RAYON_NUM_THREADS=1` for comparability. gamfit still shows 6 threads (polled), against 1 for pyGAM.

## 3. Results

The data files are `n1e3.jsonl`, `n1e4.jsonl`, `n1e5.jsonl` and `n1e6.jsonl`.
- **Complete:** the n=1e3 Gaussian block, n=1e4 Gaussian, and n=1e5 Gaussian p1 and p5.
- **Still in progress when this was written:** the GLM blocks at n=1e4/1e5, and the n=1e6 runs.
  They are 3 fits/config × minutes each at 0.2 core.
- **Spot checks instead:** where the matrix had not reached a config, I ran 1-rep spot checks
  (marked †) with the same worker.

### 3.1 Fit CPU seconds (median [min-max], 3 reps unless noted)

| family | n | design | gamfit | gamfit_k20 | pyGAM default | pyGAM gridsearch | gamfit/pyGAM | gamfit/gridsearch |
|---|---|---|---|---|---|---|---|---|
| gaussian | 1e3 | p1 | 0.24 [0.22-0.27] | 0.25 | 0.0093 | 0.094 | 25.5x | 2.5x |
| gaussian | 1e3 | p5 | 0.72 [0.65-0.73] | 1.64 | 0.037 | 0.37 | 19.3x | 1.9x |
| gaussian | 1e3 | te | 0.67 [0.63-0.67] | 2.05 | 0.035 | 0.30 | 19.4x | 2.2x |
| gaussian | 1e3 | **p20** | **timeout 400 s wall (3/3)** | **timeout** | 0.23 | 2.75 | ∞ | ∞ |
| gaussian | 1e4 | p1 | 0.37 [0.32-0.38] | - | 0.039 | 0.40 | 9.6x | 0.93x |
| gaussian | 1e4 | p5 | 1.13 [1.13-1.16] | - | 0.30 | 2.92 | 3.8x | 0.39x |
| gaussian | 1e4 | te | 1.10 [1.09-1.12] | - | 0.26 | 2.85 | 4.2x | 0.39x |
| gaussian | 1e5 | p1 | 1.40 [1.39-1.67] | 2.16 | 0.52 | 5.03 | 2.7x | 0.28x |
| gaussian | 1e5 | p5 | 15.9 [14.3-19.8] | - | 4.78 [4.58-5.36] | 50.0 [48.1-51.9] (2 reps) | 3.3x | 0.32x |
| gaussian | 1e6 | p1 | 31.8 (1 rep) | - | 12.1 (1 rep) | - | 2.6x | - |
| binomial | 1e3 | p1 | 0.53 [0.51-0.58] | 0.75 | 0.0127 | 0.098 | 41x | 5.4x |
| binomial | 1e3 | p5 | 3.66 [3.66-4.40] | 12.2 [9.6-13.0] | 0.053 | 0.38 | 69x | 9.8x |
| binomial | 1e3 | **p20** | **timeout 400 s wall** | **timeout** | 0.48 | 4.44 | ∞ | ∞ |
| binomial | 1e4 | p1 | 3.10 [3.01-3.49] | - | 0.065 | 0.43 | 48x | 7.2x |
| binomial | 1e4 | p5 | 31.4 (1 rep; trace 29.9) | - | 0.86† | 4.59† | 37x | 6.8x |
| binomial | 1e4 | te | 30.0 (trace, 1 rep) | - | 0.72† | 4.77† | 42x | 6.3x |
| binomial | 1e3 | te | 1.59 [1.47-2.49] | 4.89 | 0.049 | 0.40 | 33x | 4.0x |
| poisson | 1e3 | p1 | 0.51 [0.50-0.63] | 0.79 | 0.016 | 0.099 | 32x | 5.2x |
| poisson | 1e3 | p5 | 3.64 [3.36-4.41] | 9.24 | 0.065 | 0.42 | 56x | 8.6x |
| poisson | 1e3 | te | 2.66 [2.52-2.91] | 6.00 | 0.080 | 0.45 | 34x | 5.9x |
| poisson | 1e3 | **p20** | **timeout 400 s wall** | **timeout** | 0.76 | 4.27 | ∞ | ∞ |
| poisson | 1e4 | p1 | 4.11† | - | 0.106† | - | 39x | - |
| poisson | 1e4 | p5 | 27.3 (trace, 1 rep) | - | 1.01† | 4.77† | 27x | 5.7x |

Wall time at the observed load is about 4-6x the CPU time. For example, binomial 1e4 p5 took
**177 s wall** for gamfit against 4.6 s CPU / about 20 s wall for gridsearch.

### 3.2 Peak RSS (driver-polled, MB, median)

| family | n | design | gamfit | pyGAM | pyGAM gs | gamfit/pyGAM |
|---|---|---|---|---|---|---|
| gaussian | 1e3 | p1 / p5 / te | 127 / 149 / 148 | 112 / 122 / 121 | 114 / 124 / 124 | 1.13-1.23x |
| gaussian | 1e3 | p20 | **1390 (plateau, never finishes)** | 160 | 187 | **8.7x** |
| gaussian | 1e4 | p1 / p5 / te | 156 / 280 / 260 | 126 / 181 / 179 | 129 / 191 / 189 | 1.2-1.55x |
| gaussian | 1e5 | p1 | 465 | 275 | 278 | 1.69x |
| gaussian | 1e5 | p5 | **1398** | 800 | 821 | **1.75x** |
| gaussian | 1e6 | p1 | **2708** (ru_maxrss 2688; 15.9 s of sys CPU) | 1587 | - | **1.71x** |
| binomial | 1e3 | p5 | 176 (k20: 264) | 122 | 125 | 1.44x |
| binomial | 1e4 | p1 / p5 | 171 / 247 | 126 / 188† | 130 / 199† | 1.3-1.36x |

The import baseline is about 107 MB for gamfit and about 112 MB for pyGAM. **gamfit uses more
memory than pyGAM in every measured config.**

### 3.3 Predict CPU (n fresh rows)

| family | n | design | gamfit | pyGAM | ratio |
|---|---|---|---|---|---|
| gaussian | 1e3 | p1 | 4.3 ms | 1.5 ms | 2.9x |
| gaussian | 1e4 | p1 / p5 / te | 31 / 41 / 35 ms | 10 / 37 / 23 ms | 1.1-3.0x |
| gaussian | 1e5 | p1 | 0.28 s | 0.11 s | 2.4x |
| gaussian | 1e5 | p5 | **0.41 s** | 0.58 s | **0.71x (gamfit wins)** |
| gaussian | 1e6 | p1 | 3.65 s | 4.43 s | **0.82x (gamfit wins)** |
| binomial | 1e3 | p1 / p5 | 55 / 52 ms | 1.6 / 4.7 ms | **35x / 11x** |
| binomial | 1e3 | **te** | **1.36 s [1.19-1.37]** | 2.5 ms | **544x** (about 1.3 ms/row) |
| binomial | 1e4 | p1 | 470 ms | 11.5 ms | **41x** |
| poisson | 1e3 / 1e4 | p1 | 4.8 / 36 ms† | 1.4 / 15 ms† | 2.4-3.4x |

Binomial predict against rows (`pred_overhead.py`, s(x), p=11):

| rows | 1 | 10 | 1e3 | 1e4 | 1e5 |
|---|---|---|---|---|---|
| CPU | 0.7 ms | 1.2 ms | 46 ms | 514 ms | 4.8 s |

That is **linear at about 48 µs/row**, against about 3 µs/row for Gaussian and Poisson.

### 3.4 Import and cold time-to-first-fit

- `import gamfit`: 0.3-0.5 s. `import pygam`: 1.2-1.8 s, most of it scipy.
  **gamfit wins by about 3-4x.**
- Cold first fit (`importtime_*.txt`, `cold_profile.py`): 58% of the Python-side time of the first
  gamfit fit is the lazy `import pandas` triggered by the table-kind probe
  (`gamfit/_tables.py:53-60, 444-463`), even when the input is a dict of numpy arrays.

### 3.5 Accuracy sanity check (rmse_mu, lower is better)

- gamfit is at least as good as pyGAM default everywhere, because pyGAM's fixed lam=0.6
  overfits: edf is 67 against 30 at Gaussian 1e3 p5, and 93 against 51 at 1e5 p5.
- gamfit is roughly at parity with gridsearch. It is better at Gaussian 1e5 p5 (0.0119 against
  0.0121) and Poisson 1e3 p1, and slightly worse at Poisson 1e3 p5 (0.2475 against 0.2081, 1 rep)
  and binomial 1e4 p5 (0.0259 against 0.0231, 1 rep).
- The speed findings below therefore do not come from gamfit fitting a harder model.

## 4. Findings

### F1 — p20 (k=40 smoothing parameters) never converges: timeout and 8.7x RSS. BLOCKER, bug

**Evidence.**
- Gaussian and binomial p20 at n=1000 time out at 400 s wall, 3 of 3 reps, for both `gamfit` and
  `gamfit_k20`. pyGAM takes 0.23 s CPU; gridsearch takes 2.75 s.
- `trace_gaussian_p20_1000.log` (n=1000, p=221, k=40):
  - Planner: `Arc` with analytic Hessian, `hessian-route choice=operator reason=large_k`
    (dense_workspace 32 MB).
  - Summary line:
    `[OUTER summary] matrix-free TR finished status=MaxIterations in 200 iters elapsed=274s final_value=8.598121e2 | steps accepted=184 rejected=16 boundary_limited=184/184 radius=[3.125e-2, 2.000e0]`
  - It then warns `hit max_iter=200 ... |g|=2.620e0` and restarts seed 1 of 3, which was still
    running after 13 minutes. **Every accepted step hits the trust-region boundary, and the radius
    never exceeds 2.** This is a crawl, not convergence.
  - Cost per outer iteration: about 0.25 s CPU. That breaks down as a 1-iteration Gaussian PIRLS
    solve of 0.4-0.6 s wall plus an operator outer Hessian of 0.6-0.8 s wall.
- RSS reaches about 1.39 GB within 2 minutes and holds there, against pyGAM's 160 MB. That is
  about 6x the dense 221×221 working set. The 128 MiB PIRLS LRU (`pirls/mod.rs:124`) and the
  256-entry probe cache (`rho_optimizer/bridges.rs:152`) are count-bounded, not byte-bounded, and
  together with per-evaluation derivative stores they plausibly account for it. I did not pin the
  exact owner, because the `.so` is stripped.
- The developers already know about the budget exhaustion: `rho_optimizer/run_plan.rs:1560-1625`
  (#2817) says "All six runs ... burned their 200-iteration budget, 1200 outer evaluations". The
  crawl/thrash census is in `bridges.rs:3300-3350`.

- **Full 25-minute trace (killed by the 1500 s timeout, no fit produced).** Five consecutive
  matrix-free TR runs each hit `max_iter=200`:
  - seeds 0, 1 and 2 of the first sweep, then seeds 0 and 1 of the budget-exhaustion retry sweep,
    with seed 2 of the retry started;
  - boundary-limited on 181-199 of 181-199 accepted steps;
  - final V of 859.8, 834.75, 834.50, 834.39 and 834.81, with |g| of 2.62, 0.30, 0.20, 0.13 and
    0.33 respectively;
  - the radius collapsing to 3.9e-3 on the retry.
  That is about 1000 outer evaluations with no certified stationary point.

**Fix (SPEC-compliant).**
1. With k=40, the outer Hessian is only 40×40, so the `large_k` operator/Steihaug route is the wrong
   tool. Always use the dense analytic outer Hessian with ARC/Newton whenever k² is small, and switch
   routes on bytes of the n·p working set, not on k (`capability.rs:224-233, 458-531`).
2. Handle ρ→∞ asymptotes explicitly. Flat directions where a penalty drives a term to its null space
   produce exactly this boundary-limited crawl. Detect the directions where ∂V/∂ρ_j→0 with
   ∂²V/∂ρ_j²→0, and fix them at the boundary as mgcv does. That is a converged-parameter treatment,
   not a knob.
3. Grow the TR radius on a boundary-limited step with good ρ-agreement. With 184 of 184 steps
   boundary-limited, the radius should have expanded.
4. Bound every cache by bytes.

**Files:** `crates/gam-solve/src/rho_optimizer/{capability.rs,run_plan.rs,bridges.rs}`, `pirls/mod.rs:124`,
`reml/mod.rs:5097`. **Size: L.**

### F2 — GLM outer search uses BFGS with the analytic Hessian withheld: 6-10x slower than GCV gridsearch. HIGH, gap

**Evidence.**
- Binomial and Poisson are 5.4-9.8x slower than **gridsearch** (11 full fits) and 27-69x slower
  than pyGAM default, at every n and design measured (table 3.1). Wall time for binomial 1e4 p5
  is 177 s.
- `trace_binomial_p5_10000.log`: 57 BFGS evaluations ending in `OUTER_COST_STALL`, accepted through
  the widened curvature-scaled flat-valley bound.
- `optimizer.rs:591-605` (`standard_reml_search_prefers_gradient_only`) routes every non-identity
  link to gradient-only BFGS. Gaussian/identity uses ARC with the analytic Hessian and needs far
  fewer evaluations.

**Fix.**
- For canonical links (logit, log), the LAML outer Hessian is available analytically. It is already
  computed post-fit (F4), so use ARC/Newton for them too, which should cut outer evaluations by
  about 3-5x.
- Keep BFGS only where the third-derivative terms are not implemented.

**Files:** `crates/gam-solve/src/estimate/optimizer.rs:575-605`, `rho_optimizer/capability.rs:458-531`. **Size: M.**

### F3 — Multi-seed cascade re-solves the same optimum. MED-HIGH, gap

**Evidence.**
- Gaussian: all 3 ARC seeds converge to the same ρ̂, which is about 60% of outer time at 1e5 p5
  (`trace_g_p5_1e5.log`). The redundancy screen only fires for ρ≥0 (`seed_screening.rs:737`).
- Binomial n=3000 p5 (`trace_dbg.log`):
  - Seed 0 cost-stalls after 22 s at V=1788.132, then reports `seed 0 solver convergence claim
    failed analytic certification` (|Pg|=9.57e-3 > 3.98e-3).
  - Seed 1 restarts from ρ=1 and spends **22.7 s more** to reach V=1788.130, the same optimum
    except for a flat ρ₂ (7.14 against 11.72).

**Fix.**
- When certification fails at a stalled point, take analytic-Hessian Newton polish steps from that
  point instead of re-seeding.
- Stop the cascade as soon as a certified PD stationary point exists.
- Extend the redundancy screen to negative ρ.

This does not cut convergence short: it replaces a re-search with a second-order solve to the
same tolerance.

**Files:** `rho_optimizer/seed_screening.rs:737`, `rho_optimizer/run_plan.rs`. **Size: M.**

### F4 — Post-search work is about 40% of GLM fit time; the "CHEAP" rho-posterior diagnostic does 64 PIRLS solves on every fit. HIGH, bug

**Evidence.** `trace_binomial_p5_10000.log`: the search ends at 1m10s and the fit returns at about
2m00s, so about 40% of wall time comes after the search.
- About 4 repeated outer-Hessian evaluations at the same ρ̂ (1m18s, 1m21s, 1m37s, plus one at
  mint), about 1 s each.
- Sigma-cubature smoothing-parameter correction (1m21s-1m34s, about 11%): 8 nodes, each with up to
  9 line-search PIRLS evaluations.
- **Tier-0 ρ-posterior PSIS adequacy** (1m37s-2m00s, about 18%):
  - `rho_posterior.rs:119` has `const DEFAULT_M: usize = 64`. `rho_posterior.rs:550-600` draws
    ρ̂+L⁻¹z and runs a **full `criterion(&rho_m)` PIRLS solve per draw**.
  - The trace shows many draws failing with `Hessian not positive definite (min eig -1.6e144)` and
    gradients around 4e23 at extreme draws.
  - `optimizer.rs:3713-3737` calls it on every fit and says it is "CHEAP (a handful of
    outer-criterion evaluations) so it is emitted regardless of `skip_rho_posterior_inference`".
    It is not cheap.
  - `reml/eval.rs:970` recomputes `compute_lamlhessian_consistent` again.
- This also explains why GLM **p1** fits cost 3-4 s at n=1e4 (Poisson 4.1 s, binomial 3.1 s) for a
  2-parameter problem.

**Fix.**
- Compute diagnostics lazily. The ρ-posterior adequacy and the cubature correction only run when
  the user asks for corrected intervals or summary adequacy. This is a result-surface change, not
  a convergence change.
- Warm-start each draw from the stored IFT mode response β̂ + (∂β/∂ρ)Δρ.
- Cache the outer Hessian per ρ̂.
- Derive M from the PSIS k̂ / ESS target instead of the constant 64.

**Files:** `crates/gam-inference/src/rho_posterior.rs:119,550-600`, `crates/gam-solve/src/estimate/optimizer.rs:3713-3737`,
`crates/gam-solve/src/reml/eval.rs:970`. **Size: M.**

### F5 — Gaussian derivative evaluations re-run full n-row passes instead of sufficient statistics. HIGH, gap

**Evidence.**
- Gaussian 1e5 p5: 15.9 s CPU against 4.78 s for pyGAM default (3.3x). RSS is 1.4 GB against 800 MB.
- The trace has 253 derivative-request evaluations, each a full n-row PIRLS/assembly pass
  (`pirls/loop_driver.rs:747-795`, `newton_solve.rs:1705-1740`).
- For identity-link Gaussian, XᵀWX and Xᵀy are ρ-independent. After one O(np²) pass, every REML
  cost, gradient and Hessian evaluation should cost O(p³).
- At 1e6 p1 the fit takes 31.8 s CPU, including **15.9 s system CPU** (page-fault/allocation
  churn), and 2.7 GB RSS. Those numbers only make sense if n-sized buffers are reallocated for
  each evaluation.

**Fix.**
- For Gaussian identity with no offsets varying in ρ, form the p×p sufficient statistics once and
  route every outer evaluation (cost, gradient, Hessian) through them. The routing decision is
  cached once per fit.
- Reuse n-sized buffers across evaluations.

This is exact, not an approximation.

**Files:** `crates/gam-solve/src/pirls/loop_driver.rs:747-795`, `pirls/newton_solve.rs:1705-1740`, `reml/eval.rs`. **Size: M/L.**

### F6 — n-sized conformal substrate persisted in the model: about 617 B/row, a JSON re-parse on every accessor. HIGH (blocker at 1e6+), bug

**Evidence.**
- `model_bytes.py`: the saved model grows linearly at about **617 bytes per training row**, and is
  58.8 MB at n=1e5 for a single `s(x)`. The payload carries the full x/y "full_conformal"
  substrate.
  - Repo: `crates/gam-pyffi/.../model_payload_builders.rs:285-338`, `request.rs:690-709`. It is
    still built by default at HEAD.
- `compile_model`, `notes` and `summary` each re-parse the JSON, at 0.22-0.26 s each at n=1e5
  (installed `_model.py:129,1189`, `_api.py:1288-1299`).
- This is a large part of the RSS losses in table 3.2 (1.69-1.75x at 1e5). By extrapolation it is
  about 0.6 GB of JSON text at 1e6, plus the parsed copies. That matters under the SPEC rule that
  the program must never run out of memory.

**Fix.**
- Persist nothing n-sized by default. Conformal intervals need only the calibration residual
  quantiles (O(1)), or an explicit opt-in to keep the data.
- Use a binary, not JSON, payload for the numeric blocks, and decode it once into a Rust handle
  held by the Python object.

**Files:** `model_payload_builders.rs:285-338`, `request.rs:690-709`, `gamfit/_model.py`. **Size: M.**

### F7 — Binomial predict is 11-544x slower than pyGAM: an exact logit-normal mean plus an always-on adaptive quadrature cross-check. HIGH, bug

**Evidence.**
- Binomial predict costs about 48 µs/row; it is 35x slower than pyGAM at 1e3 rows and 41x at 1e4.
- For `te(x0, x1)` binomial it is much worse: 1.36 s [1.19-1.37] for 1e3 rows, about 1.3 ms/row
  against pyGAM's 2.5 ms total, which is **544x** (3 reps). I did not trace why te costs 27x more
  per row than s(x). My guess is that te predictions carry a larger posterior σ, which pushes rows
  onto the 160-term erfcx series plus the adaptive cross-check.
  Poisson and Gaussian cost about 3 µs/row.
- `crates/gam-solve/src/quadrature.rs:1127-1150`: `logit_posterior_meanwith_deriv_controlled`
  computes the erfcx series candidate (up to `LOGIT_MAX_TERMS=160`, line 454), then **always**
  runs `logit_posterior_meanwith_deriv_quadrature`. That is two `integrate_normal_adaptive` calls,
  for the mean and its derivative, run on every row as a drift cross-check.
- For σ<`LOGIT_ERFCX_SIGMA_MIN=0.25` (line 299), which is the usual case for predictions, the code
  goes straight to adaptive quadrature.
- The derivative is computed at predict time even though predict never uses it.
- In addition, predictions return to Python as JSON text: `model_ffi.rs:1692` (`predict_table -> PyResult<String>`)
  and `manifold/geometry_ffi.rs:7189-7206` (`serde_json::to_string(&PredictionPayload{...})`).

**Fix.**
1. Move the drift cross-check into tests or a debug assertion.
2. For small σ, use the certified asymptotic expansion E[expit(η)] ≈ expit(μ) + ½σ²expit''(μ) + O(σ⁴)
   with an explicit error bound, or a fixed-order Gauss-Hermite rule whose error bound is checked.
   Either is deterministic and exact to tolerance.
3. Skip the derivative on predict-only calls.
4. Return a numpy buffer, not JSON.

**Files:** `crates/gam-solve/src/quadrature.rs:299,454,1127-1150`, `crates/gam-pyffi/src/model/model_ffi.rs:1692`,
`crates/gam-pyffi/src/manifold/geometry_ffi.rs:7189-7206`. **Size: S/M.**

### F8 — Small-n fixed overhead: 19-25x pyGAM default and about 2-2.5x gridsearch at n=1e3 (Gaussian). MED, gap

**Evidence.**
- Gaussian at n=1e3: p1 takes 0.24 s against 0.0093 s (default) and 0.094 s (gridsearch); p5
  takes 0.72 s against 0.037 s and 0.37 s; te takes 0.67 s against 0.035 s and 0.30 s.
- At n=1e4 gamfit already beats gridsearch (0.39-0.93x). The loss is a fixed cost: 3-seed ARC
  (F3), post-fit diagnostics (F4), and payload build and parse (F6).

**Fix:** mostly follows from F3, F4 and F6. **Size: S** once those land.

### F9 — Python-layer per-call overhead. LOW-MED, gap

**Evidence.**
- `gamfit/_tables.py:53-60, 444-463`: `detect_table_kind` calls `importlib.import_module` for
  pandas, polars and pyarrow on **every** fit and predict. With polars and pyarrow missing this
  costs about 3 ms per call (17 ms over 5 predicts), and a cold first fit spends 58% of its Python
  time importing pandas for dict input.
- `_tables.py:254-280`: `_infer_categorical_from_values` runs
  `any(isinstance(v, str) for v in values)` per column, which is O(n·p) interpreted Python
  (about 5130 isinstance calls per 5×1000-row predicts). Numeric columns themselves go through
  `np.column_stack` (lines 126-140) and are fine.

**Fix.**
- Probe `sys.modules` / `type(data).__module__` instead of importing.
- Check the dtype (`np.asarray(col).dtype.kind`) before any per-element scan, or move the
  categorical inference into Rust.

**Size: S.**

### Already better (keep)

- **A1: import.** 0.3-0.5 s against pyGAM's 1.2-1.8 s.
- **A2: Gaussian vs gridsearch at n≥1e4.** gamfit is 0.28-0.39x the CPU of gridsearch, with equal
  or better rmse_mu, while doing REML with certified convergence.
- **A3: Gaussian p5 predict at 1e5.** 0.41 s against 0.58 s.
- **A4: accuracy.** rmse_mu is never worse than pyGAM default, and is at parity with gridsearch.
- **A5 (partial): no BLAS thread explosion.** With the env pinned to 1, pyGAM ran 1 thread and gamfit 6 (polled max). gamfit keeps a few helper threads even with `RAYON_NUM_THREADS=1`, but it does not spawn nproc BLAS threads per process.

### pyGAM slop to avoid (do not copy to "win" benchmarks)

- **S1: GCV/UBRE grid search.** 11 full refits on a fixed log grid, not a converged optimum. It is
  10x the cost of pyGAM default and still coarse.
- **S2: fixed lam=0.6 default.** Its speed comes from never estimating smoothness: edf 67-93 at p5,
  overfitting. Do not add a "fast mode" with fixed λ.
- **S3: unbounded OpenBLAS threads.** pyGAM runs `nproc` BLAS threads per process, which thrashes on
  shared hosts. We measured with all threads pinned to 1.
- **S4: no memory guard.** pyGAM's dense n×p model matrix is 800 MB at 1e5 p5. gamfit should beat
  this with streaming/chunked XᵀWX (F5, F6), not match it.

## 5. Open measurement gaps

- n=1e3 is complete: 144 configs × 3 reps. p20 timed out for all three families, with both gamfit variants.
- n=1e5 and 1e6 GLM, and n=1e5 te, were still running.
- At 0.2 core per process, a single gamfit binomial fit at 1e5 p5 is expected to take more than
  25 min wall.
- n=1e6 has Gaussian p1 only (1 rep each). gamfit takes 31.8 s CPU and 2.7 GB; pyGAM takes 12.1 s CPU and 1.59 GB, with gamfit's rmse_mu better (0.0018 against 0.0024). The Gaussian p5 and Poisson p1 runs at 1e6 were still going.
- The record files are appendable, and `analyze.py` regenerates every table from them.
