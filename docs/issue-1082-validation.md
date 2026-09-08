# Issue 1082 validation

Acceptance requires the original quality population to complete within 360 seconds
per test and pass its convergence, recovery, calibration and reference assertions.
A fit returning promptly, an empty test module, or a missing reference environment
does not establish completion.

The `issue-1082` nextest profile selects the original module population, the NB
outer-loop and truth-recovery regressions, and the penguin multinomial follow-up.
It runs one test process at a time, reports at 60-second intervals, terminates at
360 seconds, and writes `issue-1082.xml`. Limit Rayon and reference-library thread
pools to the allocated CPU capacity when invoking it:

```sh
cargo nextest run --profile issue-1082 --build-jobs 4 --test-threads 1 \
  -p gam --test quality_1082 \
  -p gam-models --test quality_multinomial_penguins_2612
```

## Evidence collected on 2026-09-08

- The GitHub issue is open. Its June comments explicitly leave single-fit solver
  performance unresolved; the latest existing CI ledger says the workspace quality
  population was not measured. Those records cannot establish closure.
- The ordinary nextest profile permits 600 seconds, so its success would not by
  itself meet this issue's original 360-second requirement.
- The logistic confidence-band, beta-logistic link and Cox-like marginal-slope
  files were reduced to comments by `c0a21b554`. Their module declarations still
  exist, but they contribute no tests. Replacement tests now exercise current
  production fit/prediction APIs; they remain unverified until the focused run.
- An exploratory warm-binary multinomial smooth-by-factor run on MSI acn112,
  with four Rayon workers and one BLAS/OpenMP worker, completed in **101.14 s**
  and failed accuracy: probability RMSE **0.3259004748434**, VGAM RMSE
  **0.07435029239271**. Every reported smoothing weight was
  **162754.79141900392**. This identifies an over-smoothing symptom for diagnosis,
  not current-source verification: the existing binary's exact source snapshot
  has not been certified. An earlier attempt reached the missing-VGAM reference
  failure in 98.86 s.
- The documented MSI R setup was stale. The available newer R installation and
  existing user library provide mgcv and VGAM. scoringRules, gamlss and its
  distribution package, pyGAM, and lifelines were installed for the comparisons.
  INLA's default binary requires a newer glibc than MSI. Its official Rocky
  Linux 8 binary installs and starts; the full R reference still needs its
  subprocess library-path failure resolved.
- A root build exposed four bare test `unwrap()` calls and unused parameters in
  newly added information-derivative hooks. The changes validate block/direction
  contracts, provide contextual unsupported-derivative errors, and explain test
  invariants. The scanner itself is unchanged.
- A later compile exposed an inconsistent remote GPU API snapshot. Validation
  must use the complete current local source manifest, not an overlay inferred
  from a handful of matching files.
- The complete source snapshot compiled with the scanner enabled in 9m14s.
  The dedicated profile discovered **28 tests across three binaries**. All
  three restored modules contribute an executable test.
- First census: **18 passed, 8 failed, 2 timed out**, 409.514s total. No timeout
  was increased. The command accidentally used nextest's `-j 4` (four test
  processes), rather than Cargo's build-worker meaning. Each test had four
  Rayon workers; subsequent acceptance runs explicitly use the flags above.
- Restored logistic-band coverage passed in **11.724s**. Gaussian 90% coverage
  passed in 7.827s; NB outer-loop performance passed in 2.025s; the longest
  passing test was the frailty likelihood recovery at 32.341s.
- Three tensor tests fail in covariance integration at the old fixed +/-30
  rho box (Poisson/mgcv synthetic, Poisson/pyGAM synthetic, NB/mgcv real data).
  Multinomial smooth-by-factor fails because its family floor exceeds a
  cancellation-sized resolvability ceiling. Beta-logistic fitting stops with
  projected gradient 0.6864; Cox-like marginal fitting requests a missing fifth
  likelihood derivative. INLA fails in its external binary. The R-free Poisson
  recovery fixture generates uniforms in (0,0.5] due to a 33-bit shift but
  divides by 2^32, so its labels do not follow the claimed Poisson model.
- Both penguin follow-up arms timed out at 360s. All these failures remain open
  until the corresponding production/fixture corrections pass validation.
- Six domain tests pass in 0.04s after correcting overlapping-penalty null-space
  profiling and taking the union of domains for a shared coordinate. This is
  unit-level evidence; the end-to-end rebuild is still in progress.

## Completion evidence still required

Latest focused census on MSI acn112: **24 passed, 2 failed, 1 timed out**
across 27 cases, one test process at a time with four Rayon workers. The separate
penguin follow-up is still outstanding. Synthetic Poisson tensor comparisons
passed in 5 seconds each; restored beta-logistic recovery passed in 6 seconds.
The corrected R-free Poisson fixture passed in 10 seconds and the negative-binomial
outer-loop reproduction in 2 seconds. Gaussian/logistic coverage passed in 8/12
seconds. All original recovery assertions and the 360-second deadline remain.

Remaining failures: multinomial smooth-by-factor timed out at 360 seconds;
the real spatial INLA comparison took 26 seconds but GAM's RMSE .79904 exceeded
1.10 times INLA's .70544; survival reached an inner-convergence refusal in 60
seconds. INLA now executes using its official Rocky Linux 8 binary and a real
libatomic file inside its library directory, accessible from MSI's R container.
The new survival fifth derivative passed its independent finite-difference check
against the fourth derivative, covering events, censoring, and three slope values.

The `quality_1082` target owns the selected source modules exactly once; the large
`quality` and `regressions` binaries no longer register them. Reference-quality CI
discovers and runs both quality binaries and times out focused cases at 360 seconds.
The census logs are at
`/projects/standard/hsiehph/sauer354/issue1082-focused-results/` on MSI, with the
result ledger at `/Users/user/gam-validation-artifacts/issue1082-focused-results.tsv`.

1. Run every selected test after the corrections, including both synthetic and real-data arms. Record
   actual durations and assertions; missing references remain failures.
2. Diagnose and fix remaining solver/covariance failures without increasing
   budgets, reducing the recovery population, or weakening assertions.
3. Verify affected derivative and family suites, commit and push the validated
   changes to main, then post the measured results and close the issue.
