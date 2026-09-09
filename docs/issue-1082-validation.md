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
`$MSI_HOME/issue1082-focused-results/` on MSI, with the
result ledger at `/Users/user/gam-validation-artifacts/issue1082-focused-results.tsv`.

The subsequent rounding-correction run passes the new joint-penalty cancellation
unit test (0.02s). Its multinomial inner solves now certify in a few iterations,
but repeated outer BFGS searches still reach 360s. The survival run likewise
repeats BFGS line-search failures with gradient about .426; it was stopped at
182s after exceeding its 120s fit assertion, and is recorded as terminated
(exit 143), not as a completed quality result. The next build removes the
custom-family driver's forced gradient-only preference, allowing the outer
planner to use the family's declared exact Hessian.

The Poisson regression also passes production posterior-mean and uncertainty
checks (9.41s): RMSE .0890 against the original .3634 bar, EDF 20.17, all 4500
truth means inside the nominal 95% bands, RMS standardized error .5824. Commit
`3a97e147a` retains both the original recovery bars and the uncertainty checks.

Production posterior prediction leaves the spatial discrepancy essentially
unchanged (RMSE .79949). An independent mgcv tensor with the same 7×7 basis gives
.80337; one using the existing INLA mesh's twelve-interval spatial resolution
(13 value knots per tensor margin) gives .74762. The revised comparison ties
both representations to that shared resolution; its original R² and relative
RMSE requirements remain unchanged. The fresh GAM run passes in 65.30s:
posterior-mean RMSE .73052 versus INLA .70544 (ratio 1.036), R² .8767, EDF 33.11
within the 169-column tensor basis.

Final curvature-enabled rerun: multinomial still timed out at 360 seconds.
Survival's fit converged in 132.20 seconds, but failed its unchanged 120-second
fit assertion (133 seconds total); its later prediction-quality assertions were
not reached. The MSI connection closed during the first penguin arm, leaving
both fresh penguin results incomplete. The compute nodes subsequently did not
answer. Logs are in `issue1082-curvature-results/` beside the earlier census.

The attempted small multinomial outer-derivative probe failed immediately because
the public joint-hyper evaluator counted block-local penalties while this family
uses full-width joint penalties (six supplied rho coordinates versus zero
expected). This is a harness failure, not derivative evidence. The unusable probe
was removed from the suite and saved outside the repository for further work;
the previously comment-only module supplies no coverage. The independent
joint-penalty cancellation unit check passed in 0.02 seconds. Issue #1082 remains
open: neither the outstanding quality gates nor all affected derivative checks
have been satisfied.

On the next availability check, the login node and primary Slurm controller
responded, but `sinfo` reported 490 invalid, 11 unknown and one draining node;
neither default compute host answered. No scheduler jobs remained for this user.
No replacement validation job was submitted into that unavailable pool.

Recovered multinomial logs narrow the remaining performance investigation:
at `n=270, p=38, k=24`, a late armed-Jeffreys evaluation spent 1.802 seconds
assembling its dense outer Hessian out of 1.975 seconds total. Adjacent inner
solves certified in 0.034 and 0.047 seconds. Source inspection finds repeated
first and all-axis second information derivatives for each pair in
`custom_family_outer_jeffreys_hphi_drift_batched`; the mixed third derivative is
pair-specific. This identifies work to measure, not proof of a derivative error
or a validated optimization. The failed public joint-hyper probe must also be
replaced by a check of the labeled joint-penalty evaluator actually used by the
production fit. A local copy of the 5.2 MB failure log is preserved at
`/Users/user/gam-validation-artifacts/issue1082-curvature-multinomial.log`.

acn112 subsequently rebooted and became reachable again. A focused rebuild of
the existing snapshot completed in 67 seconds. The corrected derivative probe
passes joint strengths through `JointPenaltyBundle` and supplies an empty
block-local rho vector, as the production labeled evaluator does. It now reaches
the derivative assertions: **one pass, one failure in 0.26 seconds**. Without
Jeffreys/Firth, all six gradient coordinates and the full Hessian pass at both
tested rho points. With the term active, the first gradient coordinate is
0.4309032880370025 versus central finite difference 0.4305678388227818 at
rho `[-.75, -.45, -.15, .15, .45, .75]`; this exceeds the existing relative
1e-5 test bar. Its Hessian assertion is not reached. This narrows the remaining
correctness investigation to the armed criterion; step-size stability and the
responsible derivative term still need investigation. The failing reproduction
is retained in the worktree at
`crates/gam-models/tests/multinomial_outer_derivatives_1082.rs`, and its MSI log is
`$MSI_HOME/issue1082-resume-derivatives.log`. No further
long quality run was launched after the request to finish immediately.

1. Run every selected test after the corrections, including both synthetic and real-data arms. Record
   actual durations and assertions; missing references remain failures.
2. Diagnose and fix remaining solver/covariance failures without increasing
   budgets, reducing the recovery population, or weakening assertions.
3. Verify affected derivative and family suites, commit and push the validated
   changes to main, then post the measured results and close the issue.
