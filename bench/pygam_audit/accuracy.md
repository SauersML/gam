# Accuracy audit: gamfit vs pyGAM, out-of-sample, head to head

**Axis:** out-of-sample predictive accuracy.

**Versions tested:**
- gamfit 0.1.267, the installed wheel. HEAD is at c124637, and pyproject says 0.1.268.
- pyGAM 0.12.0.
- Nothing under /home/user/gam was modified or built.

**What was compared.** Every case uses 5-fold CV: shuffled KFold, with StratifiedKFold for binomial, seed 0. Each fold was fitted three ways:
- `pygam_default`: `LinearGAM`/`LogisticGAM`/`PoissonGAM`/`GammaGAM` at the defaults (n_splines=20, lam=0.6).
- `pygam_grid`: pyGAM's `gridsearch()`, its intended use (a shared lam over logspace(-3,3,11), GCV/UBRE).
- `gamfit`: `gamfit.fit(data, formula, family=...)` using the default formula equivalent (REML/LAML, posterior-mean prediction), with no tuning.

**Metrics.** Held-out deviance/MSE, logloss, RMSE, and Brier/AUC for binomial. Synthetic cases also get truth MSE, which is mean((mu_hat − mu_true)²) on the held-out rows.

**Verdict rule.** Paired per-fold difference d = gamfit − pyGAM. WIN if mean d < −2·SE, LOSS if mean d > +2·SE, otherwise TIE. Any gamfit fold that raises is a LOSS(fail), because SPEC says a fit only comes from a converged optimization.

## Harness (permanent benchmark)

Everything is under `scratchpad/audit/accuracy/`:

| file | purpose |
|---|---|
| `bench_accuracy.py` | The harness. Case catalogue: pyGAM datasets (`pygam.datasets`, cached in `pygam_data/`), the repo CSVs in `bench/datasets`, and synthetic truths from n=100 to 50k. Results go to `results/<case>.json`, one file per case, and the run is resumable. No timing is recorded. |
| `report_accuracy.py` | Aggregates the results into the tally, loss list, per-case tables and error list (`results_tables.md`). |
| `probe_variant.py` | Reruns gamfit on the same folds with a formula variant (`default`, `dp0`, `kNN`, `f:<formula>`). Prints paired differences against the default, plus the stored pyGAM means. |
| `repro_binom300.py`, `repro_nearsep.py`, `repro_wage.py`, `repro_year.py` | Deterministic repros of the fit failures. |
| `probe_te.py`, `probe_lowcount.py`, `probe_1680.py`, `probe_k.py`, `probe_kcheck.py`, `probe_coal.py`, `laml_coal.py` | Diagnosis probes. |

Usage:
```
cd scratchpad/audit/accuracy
../../bvenv/bin/python bench_accuracy.py --list                 # case catalogue
../../bvenv/bin/python bench_accuracy.py --skip 'n50000|n20000' # everything but the big cases
../../bvenv/bin/python bench_accuracy.py --only '^g1d_sin6' --force
../../bvenv/bin/python report_accuracy.py --md results_tables.md
../../bvenv/bin/python probe_variant.py '^bike$' default 'f:y ~ te(season, hour, k=[4,12])'
```

Note: the g1d synthetic data seed changed partway through the audit. It was `hash()`, which Python salts per process, and it is now `zlib.crc32`. The stored g1d results were produced with hash-seeded draws of the same truth, noise level and n, so the conclusions do not depend on the seed. From now on the benchmark is reproducible byte for byte. `probe_variant.py` does not print stored pyGAM numbers for g1d cases.

**Coverage.** 55 cases are complete. The following were still running or unfinished when this report was written:
- toy_classification, chicago and chicago_docs (pyGAM datasets)
- toy_interaction
- heart_failure, quakes, quakes_space, badhealth and penguins_mass (repo CSVs)
- the n=20k/50k cases, which were deliberately not run because the CPU is shared

They can be finished with `bench_accuracy.py`, which skips completed cases.

## Tally (55 cases)

| comparison | metric | WIN | TIE | LOSS (incl. fail) |
|---|---|---|---|---|
| vs pyGAM default | held-out | 9 | 31 | 15 |
| vs pyGAM default | truth MSE (synthetic) | 20 | 3 | 15 |
| vs pyGAM gridsearch | held-out | 8 | 26 | 21 |
| vs pyGAM gridsearch | truth MSE (synthetic) | 12 | 10 | 16 |

Every LOSS falls into one of four causes:
- **(A) The default basis is too small.** This is most losses.
- **(B) Fit failures raised by the installed wheel**, either a cubature IntegrationError or an outer optimizer that did not certify.
- **(C) Two small statistical trade-offs:** low-count Poisson, and Gaussian with outliers at n=300.
- **(D) A 0.2% gamma deviance difference** where gamfit wins truth MSE by 35%.

Full tables are in `accuracy/results_tables.md`.

## Losses, diagnosed

### A. The default smooth basis is too small and does not grow with n (blocker)

`crates/gam-terms/src/term_builder.rs`:
- `MAX_DEFAULT_INTERNAL_KNOTS = 8` (around lines 4478-4514). `heuristic_knots_for_column` returns `(unique/4).clamp(4, 8)`, so a default `s(x)` has at most 12 cubic B-spline functions at every n.
- `heuristic_tensor_margin_knots` (around 4550) gives te margins of about 7 each, also capped by `unique/4`. For hour (24 unique values) that is 6.

pyGAM uses 20 functions per smooth and 10×10 for te. Once n is large enough to resolve the truth, the gamfit cap is a bias floor that data cannot fix.

In the table below, "(dev)" means held-out MSE, which is the same thing as held-out deviance for Gaussian.

| case | gamfit default | pyGAM grid | gamfit with larger k (same folds) |
|---|---|---|---|
| g1d_sin6_n500, truth MSE | 0.4537 (edf 11.0/12) | 0.00797 | k=20: **0.0078** |
| g1d_sin6_n2000 | 0.4061 | 0.00435 | |
| g1d_sin6_n10000 | 0.4067 (flat in n, +10,700%) | 0.00376 | |
| g1d_sin3_n2000 | 0.00200 | 0.00070 | k=20: **0.00144**, d=−0.00099±0.00011 |
| g1d_sin3_n10000 | 0.00170 | 0.00014 (+1080%) | |
| g1d_doppler_n500 | 0.283 | 0.190 | k=20: 0.155, d=−0.081±0.007 |
| g1d_doppler_n2000 | 0.249 | 0.152 | |
| bump2d_n4000, te, truth MSE | 0.00794 (margins [7,7], edf 47/48 cols) | 0.00230 | te k=10: **0.00222**; k=8: 0.00249 |
| bump2d_n1000, te | 0.0105 | 0.0062 | |
| bike (te(season,hour)), held-out dev | 0.167 (edf 33) | 0.0673 | te k=[4,12]: **0.0525**; k=[4,20]: **0.0511**; k=10: 0.0587 (d≈−0.11±0.008, beats the grid by 25%) |
| head_circumference (s(age), n=7040), dev | 2.834 (edf 11.9) | 2.771 | k=20: **2.7708** (d=−0.063±0.0095, ties the grid) |

**The engine already knows.** The basis-adequacy check, #2774 in `crates/gam-models/src/fit_orchestration/drivers/basis_adequacy.rs`, fires on these fits:
- bump2d te: "lack-of-fit p = 7.7e-22 against 189 higher-resolution directions".
- g1d_sin6 n=500: p=9.3e-41.

It only warns, and returns the underfit model.

**Larger bases are safe under REML with the function penalty**, measured on the same folds:

| check | default | larger basis |
|---|---|---|
| null3_n200, truth MSE | 0.0449 | k=20: 0.0453 |
| null3_n2000, truth MSE (null still recovered, edf 1.5) | 0.00065 | k=20: 0.00064 |
| add4_n1000, truth MSE | 0.105 | k=20: 0.099 |
| binom_add4_n1000, truth MSE | 0.00434 | k=20: 0.00421 |
| real data (mcycle, coal, faithful, hepatitis, lidar, gagurine) | | k=20: all within noise |

**The #1680 rationale quoted in the cap's comment does not reproduce on this solver.** The comment claims a 24-function basis gave RMSE 0.39 vs 0.12. I re-ran the design from `tests/regressions/misc/bug_hunt_1680_near_collinear_additive_recovery.rs` (`probe_1680.py`: n=120, x2/x3 at ρ=0.985, 4 seeds). Mean truth RMSE was:

| design | default | k=16 | k=24 |
|---|---|---|---|
| collinear | 0.098 | 0.099 | 0.100 |
| independent | 0.095 | 0.095 | 0.096 |

**Costs of k=20, all small:**
- hetero_n500: 0.0116 → 0.0138, +19%. It stays below pyGAM grid's 0.0140 and just above pyGAM default's 0.0129.
- pois_add2_n300: +0.087±0.046, not significant.
- gamma_add2_n300: +0.012±0.007, not significant.

**Fix (M), SPEC-compliant: no grid, no knob, REML still decides the smoothness.**
1. Raise the default 1-D basis. Replace the flat cap of 8 with a cap that grows with the data: for example `clamp(unique/4, 4, 20..35)`, as in Ruppert's `min(unique/4, 35)` rule, or at least 20 functions.
2. Grow te margins the same way: about 10 per margin for d=2, and let a margin like hour (24 unique) use up to about unique−1 rather than unique/4.
3. Rewrite the comment block at term_builder.rs:4484-4514. Keep the #1680 regression test as the guard; it passes at k=24 today.
4. Optionally, make the #2774 lack-of-fit test act rather than warn. When it rejects decisively, refit with that smooth's basis doubled, and repeat until the test accepts. This is test-driven basis enrichment, not a search over λ. The file is `crates/gam-models/src/fit_orchestration/drivers/basis_adequacy.rs`.

Add g1d_sin6, bump2d and bike from this harness as accuracy regressions, each with a truth-MSE bound.

### B1. Cubature IntegrationError raised on ordinary data (high; fixed in HEAD, not released)

"smoothing cubature has no positive-width proposal in the resolved domain" fails these folds:
- wage: folds 2 and 3. gamfit wins the 3 folds it completes (1515/1282/1022 vs pyGAM 1518/1289/1033).
- cake: folds 0 and 3.
- pois_add2_n300: fold 3.
- pois_add2_n2000: fold 4. The covariates are continuous uniform, so this is not only a low-cardinality issue.
- nearsep_n200: fold 0, where the Firth rescue was also attempted and failed.
- city_temp: 2 folds at k=30.
- haberman at k=20, in the variant "…proposal has no positive width".

The error is raised at `crates/gam-solve/src/reml/eval.rs:1379`. HEAD catches it and falls back to first-order numerical integration (eval.rs:1411-1419, commit 5340cb4, which bumped pyproject to 0.1.268). `strings` on the installed 0.1.267 `.so` finds no match for the fallback text.

**Fix (S).**
- Release 0.1.268.
- Add regression tests built from these folds. The npz files and repro scripts for wage are here; the cake and pois_add2 folds are selectable with `bench_accuracy.py --only`.
- Also check the second message variant, "proposal has no positive width", which appeared in haberman k=20 fits. It may be a separate code path that the HEAD fallback does not cover.

### B2. The outer REML optimizer fails to certify stationarity (high)

- binom_add4_n300, fold 0 (`repro_binom300.py`, deterministic): |Pg|=1.456e-3 > bound 1.343e-3, termination=line_search_failed.
- nearsep_n200, fold 2: |Pg|=8.5e-3 > 2.2e-4.
- haberman, folds 2 and 4:
  - fold 2: |Pg|=7.5e-6 > 3.65e-6, hessian_psd=NO.
  - fold 4: |Pg|=6.99e-3 > 3.65e-6.

pyGAM "succeeds" on all of these only because it never checks convergence. gamfit refusing to ship an uncertified fit is correct per SPEC. The accuracy loss is still real, since the user gets no model.

The messages come from the `crates/gam-solve/src/rho_optimizer/` paths ("did not certify", and "tail-snap declined" at run.rs:5752), which are unchanged in HEAD. I could not verify HEAD's behaviour without a build.

**Fix (M/L), in the opt crate.**
- When the BFGS line search fails near a stationary point, switch to an analytic-Hessian Newton or trust-region step. The code already computes the terminal analytic curvature.
- Add eigenvalue-shifted steps for the indefinite case (hessian_psd=NO), rather than terminating.
- Use these three repros as regression tests.

### C1. pois_lowcount_n500: truth MSE +33% vs pyGAM default (low/med; tie vs grid)

| fit | truth MSE |
|---|---|
| gamfit default | 0.00690 |
| gamfit k=20 | 0.00690 (identical) |
| gamfit double_penalty=false | 0.00553 (d=−0.00137±0.00024) |
| pyGAM default | 0.00519 |
| pyGAM grid | 0.00568 |

This is not basis-driven. The loss comes from the null-space ridge of the double penalty shrinking the linear part at low counts. dp0 is not the fix: `probe_dp0.out` shows it costs the null cases badly (null3_n200 0.045→0.062; null3_n2000 0.00065→0.00186) and costs add4, binom_sin2 and gamma_add2 by about 10%.

**Fix (M):** investigate the null-space λ selection in low-count Poisson LAML. Look at the LAML curvature around the null-space ρ with μ≈0.3, and whether the Laplace approximation is biased there. Keep the double penalty as the default.

### C2. outlier_n300: truth MSE 0.32 vs pyGAM default 0.21 (low; tie vs grid)

The d is about 2 SE. One fold collapses to edf 1.0: Gaussian REML with heavy-tailed contamination picks λ→∞. pyGAM's fixed lam=0.6 wins by accident.

At n=2000 gamfit wins: 0.060 vs 0.166 against pyGAM default and 0.068 against the grid.

**Fix (L):** add a scaled-t / robust family with df estimated by LAML. Neither library has one; see `crates/gam-models/src/fit_orchestration/materialize/family.rs`. This is not a default change.

### D. gamma_add2_n2000: held-out deviance +0.2% vs grid (low; not actionable)

d=0.00126±0.0003. gamfit wins truth MSE by 35% (0.059 vs 0.091), so the held-out deviance difference reflects noise in the gamma deviance and does not mean the grid fit is closer to the truth. No action.

### Examined and not a real gap

- **coal (Poisson):** TIE, 1.175 vs 1.131. It is a genuinely flat LAML optimum, and one fold goes to edf 2 (`probe_coal.py`, `laml_coal.py`). An optional improvement (M) is to integrate the predictive mean over the ρ posterior when the LAML is flat.
- **city_temp:** TIE (11.01 vs 10.52 ± 0.43·2). k=20 and k=30 do not help.
- **hepatitis, prostate, default (credit), faithful, nottem, mcycle:** TIE.

## Already better than pyGAM, confirmed

| case | gamfit | pyGAM default | pyGAM grid |
|---|---|---|---|
| null3_n2000, truth MSE | 0.00065 (edf → 1.5) | 0.0228 (−97%) | 0.0033 (−80%) |
| add4 n200/1000/5000, truth MSE | 0.49 / 0.105 / 0.0144 | −51% / −48% / −72% | −32% / −49% / −64% |
| binom_add4 n1000/5000, truth MSE | | −53% / −68% | −45% / −60%; AUC and Brier also win |
| gamma_add2 n300/2000, truth MSE | | −59% / −67% | −38% / −35% |
| hetero, nearsep_n1000, outlier_n2000, bump2d_n1000 (vs default), pois_add2_n10000 | wins or ties | | |
| trees (Gamma, n=31), dev | 0.0073 | 0.45 | 0.71 |
| sleepstudy, dev | 1079 | 2028 | 2106 |
| gagurine, lidar | win | | |

In trees, pyGAM fits 40 coefficients (edof ~21 on 25 training rows). In sleepstudy, pyGAM penalizes the factor term.

## pyGAM choices gamfit must not copy

- **Fixed lam=0.6 default.** It "wins" outlier_n300 and pois_lowcount by accident and loses badly elsewhere: null3_n2000 truth MSE 35x, add4 about 2x.
- **gridsearch.** One shared λ for all terms, chosen by GCV over 11 grid points. It over-smooths some terms and under-smooths others (add4: −49% for gamfit). It also overfits tiny data: trees edof 21 at n=25.
- **Penalized factor terms** (sleepstudy, 2x worse).
- **No convergence certification.** pyGAM returns a model on every fold where gamfit raises. That is the right behaviour for gamfit under SPEC, but the underlying optimizer robustness still has to be fixed (B2).

The one pyGAM choice worth matching is a basis big enough for the data (20 per smooth, 10×10 for te), with REML still doing the smoothing (finding A).
