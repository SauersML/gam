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

## Evidence collected on 2026-09-11

All measurements below are at `origin/main` `73f25df20`, test profile (optimized
workspace crates), four Rayon workers, one test process per case, 400-second job
cap against the unchanged 360-second acceptance line. Logs are under
`/scratch.global/sauer354/i1082-logs/` on MSI.

**Reference stack.** MSI's `R/4.2.2-openblas` module does not start on the
msismall nodes (`libreadline.so.6` missing; R exits 127) and the system python is
3.6.8 without lifelines or pyGAM, so the first census (job 381385) measured
timing only. A working stack is
`PATH=/common/software/install/manual/R/4.2.0-openblas-rocky8-fix/bin:/scratch.global/sauer354/w4-pyref310/bin`
with `R_LIBS_USER=/scratch.global/sauer354/w4-Rlib`: mgcv, VGAM, nnet, gamlss and
survival load, and the python environment carries lifelines 0.30.0 and pyGAM
0.12.0. INLA does not load there: after the udunits module supplies
`libudunits2.so.0`, the library's `sf` and `fmesher` builds require
`GLIBC_2.32`/`GLIBCXX_3.4.29`, newer than the Rocky 8 nodes provide.

**Focused census with references (job 383949): 26 of 27 `quality_1082` cases
pass**, the slowest in 29.8 s (Cox-like survival), 25.3 s (frailty), 21.2 s
(multinomial smooth-by-factor) and 20.8 s (pyGAM Poisson real data). The 27th,
the INLA spatial comparison, is `REFERENCE_ENV_MISSING:INLA` and is not a
measurement of gam. Both penguin follow-up arms still exceed the budget: killed at
400 s (job 381385) and again at 420 s with INFO logging (job 383312).

**Penguin mechanism (job 383312, arm 1).** The unbiased probe (24 outer
coordinates, every evaluation a dense value/gradient/Hessian at 0.3–1 s; 527
evaluations summing to 249.7 s) reaches an incumbent with projected gradient
6.3e-4 against the certificate's solver-band bound 2.57e-3. The cost-stall guard
reads its reduced Hessian as a strict saddle and grants eight escapes, each buying
about 3e-5 of objective. That is above the window's relative floor, so the window
never fills and the seed runs to `max_iter = 100`. Seeds 1–3 repeat this, and the
budget-exhaustion retry replays seed 1 bit-identically (final value 9.333945e0,
33 accepted / 46 rejected, twice). The terminal certificate then withdraws the
curvature verdict — `λ_min = −6.696e−7` predicts a decrease of 3.35e−7 against a
criterion resolution of 2.525e−5 — and accepts `|Pg| = 4.06e−4`, but the probe
is refused for lack of a certified optimum on its budget and the fit arms
Jeffreys/Firth before the deadline.

**Armed outer-gradient defect localized.** The failing derivative reproduction is
a formula defect in the Jeffreys composition, not inner precision and not the
family:

- job 383758: the armed analytic gradient is identical from a cold start, from its
  own mode and from four displaced warm starts, at inner tolerance 1e-10 and
  1e-13; the inner mode certifies at residual 7.36e-11;
- job 384768: every multinomial Jeffreys information hook (first, second, all-axes
  second, all-axes third derivative) matches central differences to at most
  7e-10 relative at `h = 1e-5`;
- job 385018: the value-path Jeffreys gradient matches FD of `Φ` (≤ 1.4e-7) and
  the batched drift `D_β H_Φ[u]` matches FD of `H_Φ` (≤ 1e-9), but
  `H_Φ + completion` against `−FD(∇Φ)` is off by 1.067 and 1.080 relative in the
  moderate regime (full and intercept spans), 0.234 for the separated intercept
  span, and exact (3.4e-10) only where the conditioning gate is saturated.

With `Φ = G·U`, the exact Hessian also carries `∇G⊗∇U + ∇U⊗∇G + U·∇²G` and the
motion of the relative floor `REL·λ_max`. The mode response `∂β̂/∂ρ` is solved on
`M_true = H + S_λ + H_Φ + completion`, so without those terms it is the response
of a different objective. The correction adds
`JointJeffreysPlan::hessian_motion` and routes
`custom_family_joint_jeffreys_second_order_completion` through it; both are exactly
unchanged where the gate is saturated and no eigenvalue feels the floor. Two unit
tests compare the completed Hessian with finite differences in the gate band and
under a moving floor, each with a positive control that the frozen-policy Hessian
misses by more than 1e-3.

**Resolution-aware saddle verdict.** The dense ARC bridge and its seed now take
the reduced-Hessian definiteness verdict at the criterion's curvature resolution
`2·rel_cost_floor·(1 + |V|)` through `certificate_curvature_shift`, the
certificate's single owner of resolvable curvature. Negative curvature that one
e-fold of `log λ` cannot turn into a resolvable decrease no longer forces escapes.
Making the online curvature-stationary exit consistent with that standard, and
removing the bit-identical seed replay, belong to #2817 and are being done there.

**Validation of both corrections (job 388047, `37c192a39` + both patches).** The
workspace gate passes, as do 386 gam-solve Jeffreys/rho-optimizer pins and 16
gam-custom-family `jeffreys` pins. The armed outer gradient now matches central
differences to at most 2.0e-9 relative on all six ρ axes at shifts 0 and −3, and
smooth-by-factor passes in 25.9 s. The derivative reproduction still fails on the
Hessian, `(3,3)` analytic 0.19796545 against FD 0.19753201. The mode response is
solved on the motion-completed curvature, but its outer-Hessian drift
(`completion_beta`) still differentiated only the frozen-policy completion. Both
penguin arms are killed at 400 s again. The terminal adjudication measures the
analytic outer Hessian against the criterion's own curvature along the disputed
eigenvector: ‖dH‖ = 1.1e−6 on arm 1 (sub-resolution) but 0.272 on arm 2 (analytic
λ_min = −4.9e−5 against criterion curvature 0.295 ± 0.023). An inexact outer
Hessian is therefore still manufacturing negative curvature there.

**Outer-Hessian drift of the motion (`dc97c1b68`).**
`JeffreysHphiDriftBase::motion_drift_action` returns `D_u M[·, v]`, the β-drift of
the gate/floor motion part `M` of `∇²Φ`. It is the product rule over every factor
of `M`, with the extreme eigenvalues to third order, the gate's third partials and
the floor's third-order sensitivities, and `completion_beta` subtracts it where the
motion is active. Unit tests pin the gate's third partials (both bands) and the
gate-band and moving-floor curvature drift against central differences, each with
a positive control that the frozen drift misses the term.

1. Run every selected test after the corrections, including both synthetic and real-data arms. Record
   actual durations and assertions; missing references remain failures.
2. Diagnose and fix remaining solver/covariance failures without increasing
   budgets, reducing the recovery population, or weakening assertions.
3. Verify affected derivative and family suites, commit and push the validated
   changes to main, then post the measured results and close the issue.
