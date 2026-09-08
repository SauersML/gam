# Issue 2668: original-contract verification

The acceptance scope is every one of the issue's original 30 entries, including
the two deleted contracts. Moving failures to another issue is not completion.
The issue remains unresolved.

`tests/data/issue_2668_regressions.json` records all 30 original identities. It
explicitly maps the remaining renamed seed test. The original irrelevant-term
shrinkage test has since been restored, so its mapping again uses its original
identity. The runner resolves
each identity against the executable's actual inventory, requires exactly one
match, and requires a terminal verdict of one passed test. Missing, ambiguous,
failed, and timed-out entries cannot contribute to the pass count.

## September 8 diagnostic baseline

On MSI acn112, the existing
`/scratch.global/sauer354/m1-target/debug/deps/regressions-136c6e923b488523`
binary produced:

| Verdict | Entries |
| --- | ---: |
| Passed | 23 |
| Failed | 2 |
| Missing | 2 |
| Diagnostic timeout | 3 |

The runner used two test workers, each limited to two Rayon threads, and a
60-second per-test diagnostic cap. The binary's SHA-256, exact inventory,
individual elapsed times and exit codes are recorded in
`bench/measurements/issue_2668/issue2668-baseline/results.json`. This is an
existing-binary measurement, **not verification of the current worktree**.

The two failures are the SAE ARD pair. The missing entries are Firth inference
for loglog/cauchit and negative-binomial generation/prediction variance agreement.
The timeouts are the negative-binomial covariance/prediction fit, the replacement
double-penalty smooth test, and survival location-scale EDF. A diagnostic timeout
does not establish an assertion failure.

## Restored and strengthened tests

The Firth test again enables bias reduction and inference for both noncanonical
links. It requires finite coefficients and deviance and a correctly sized,
finite covariance with positive diagonal. The restored test and its heuristic
seed sibling passed in **2.78 seconds** against the existing compiled libraries.

The negative-binomial scenario again constructs heteroscedastic count data using
the gamma/Poisson mixture. The original per-row variance equality and nonconstant
dispersion checks remain. The restored test hit the **60-second** diagnostic cap;
it has not passed and has not been removed.

The SAE affine evaluator formerly returned the original basis table after latent
coordinates changed, despite reporting a nonzero Jacobian. It now evaluates the
specified affine function at each trial coordinate. Constant derivatives on a
finite set of observations do not prove that an unknown function is affine.

The smoothness test's finite grid missed its minimum. It now checks the exact
profiled scalar criterion, including the existing rank charge. With
`delta = beta_0 - beta_1`, `G = X'X`, and
`h = 1 / ([1,-1] G^-1 [1,-1]')`, its lambda-dependent part is

```text
V(lambda) = lambda*h*delta_hat^2 / (2*(h+lambda))
          + log(1+h/lambda)/2 + log(n)*h/(2*(h+lambda)).
lambda_star = h / (h*delta_hat^2 - 1 - log(n)).
```

Four score differences around this closed-form minimum agree within `1e-6`, and
each is positive. This stronger test passed in **0.22 seconds** against the
existing compiled libraries.

## Remaining root causes and verification

The SAE production criterion subtracts the coordinate log determinant while
retaining its Gaussian ARD normalizer. Its EFS proposal still assumes the full
coordinate determinant is present. The baseline records
`V(alpha*)=-0.5275459907111362` and
`V(e*alpha*)=-2.527545990710731`: the slope is exactly `-n/2=-2` for four rows.
The production documentation in `construction_quasi_laplace.rs` explicitly
acknowledges that the old balancing term was removed. A cap on alpha cannot
repair this mismatch between the criterion and the update equation.

The corrected affine fixture also exposes an unpenalized decoder scale orbit:
the collapse probe gives its linear decoder a zero function Gram. Shrinking the
latent coordinates and expanding the decoder preserves reconstruction while
reducing the coordinate penalty. Its claim of a finite, identified joint optimum
therefore needs a mathematical repair as well as an objective repair. The current
tests still report indefinite observed information or a precomputed-basis encode
refusal; they are not marked fixed.

Current-source Cargo verification reached the repository scanner and was refused
for source violations in other in-flight files. The first remote synchronization
also mixed a newer remote Python manifest with the local Rust manifest; that
version diagnostic is not evidence of a local defect. Focused diagnostics were
compiled directly from the edited test files against the already-built coherent
GAM libraries. No scanner was disabled, and these diagnostic passes do not
substitute for a successful current-source build and complete remeasurement.

Reproduce the inventory accounting on a successfully built current executable:

```sh
python3.11 scripts/verify_issue_2668.py /path/to/regressions /shared/path/to/receipt
```

Completion requires fixing the criterion and its derivatives together, replacing
invalid fixture premises with independently derived contracts, resolving the
inner-solver/performance failures, and rerunning every original contract on the
repaired source. No closure claim has been made.

## Current-source snapshot and gamma kernel measurement

The scanner errors were resolved in the shared source, and Cargo successfully
built `regressions-77fb4bb5c2a1941a`. Its archived 30-entry measurement is
`bench/measurements/issue_2668/issue2668-source-r2/results.json`:
22 passed, five failed, one missing mapping, and two diagnostic timeouts. The
restored negative-binomial variance test passed in 40.68 seconds; Firth passed
in 0.72 seconds. This supersedes the earlier existing-library observations for
those tests. The missing mapping was the obsolete supported-linear replacement;
the original irrelevant-covariate test is present and must be measured separately.

The additional failures were the heuristic seed and the concave scenario shared
by the CLI/FFI parity and shape tests. Diagnostics show that terminal REML
certification reset the inner accuracy history and solved the mode more coarsely
than the search. The shape fit instead enforced KKT on an unfinished inner solve,
before its non-convergence status could update the iteration-cap feedback and
reject the trial. Corrections are under verification; neither is counted fixed.

A bounded 20-second profile of the old-library negative-binomial fit attributed
13.53% of cycles to one factorial iterator symbol, plus additional factorial and
polygamma work. Gamma derivative orders are compile-time constants; their
Bernoulli/factorial coefficients now are too. The third-order dispersion path
also requests exactly four scalar derivatives rather than computing five and
discarding the last one. Series length, recurrence threshold, and arithmetic
order are preserved.

Fifteen gamma/jet correctness tests passed on MSI. The permanent
`gamma_stack_bench.rs` evaluates 12 arguments from `1e-8` to `1e8`, 10,000 times
each. Three interleaved pairs under the optimized test profile measured:

| Arm | ns per full derivative stack |
| --- | --- |
| Before | 589.0, 582.2, 582.6 |
| After | 255.3, 252.6, 311.9 |

All six output fingerprints are `41fee25e33145465`. Median speedup is 2.28×.
The before library is `gam-math-44e36aea55b719e1`, the after library is
`gam-math-dc03c8141bcd374b`; both Cargo profile fingerprints are
`16635960049555823289`. Dependency feature graphs differ between the root
regression build and the standalone math build, so this is a kernel diagnostic,
not a controlled end-to-end fit speedup. An older unoptimized library was also
measured; its much larger ratio is deliberately excluded from this comparison.
The timing receipt and textual perf report are retained beside the benchmark.

## Verified solver-contract corrections

The next scanner-clean executable measured **25 passes, three failures, and two
timeouts**, with all 30 entries present. The runner verified the binary hash
again afterward; `binary_unchanged` is true. Full receipts are in
`bench/measurements/issue_2668/issue2668-source-r4/`.

The seeded REML, CLI/FFI parity, and all-shapes regressions now pass. Full-fidelity
REML evaluations use the existing derivative-budget inner tolerance independently
of adaptive history. KKT audits apply to claimed inner minima; unfinished solves
reach the existing non-convergence handling, which rejects the trial and updates
the cap feedback. No stationarity requirement was relaxed. The solver-only seed
reproducer also passes, with terminal outer gradient norm approximately `1.68e-7`.

The three remaining failures are the SAE ARD pair and the restored
irrelevant-covariate shrinkage test. Shrinkage currently fails during smoothing
uncertainty integration (`positive smoothing cubature proposal extends beyond the
resolved rho domain`), before reaching its EDF assertion. The NB covariance fit
and survival location-scale EDF fit still exceed 60 seconds. Their new logs
capture inner/outer progress rather than merely a missing terminal line.

The shared worktree now contains a separate correction distinguishing posterior
support from the optimizer's numerical resolvability range, and using the
declared proper smoothing prior for cubature weights. That newer change is not
part of the r4 measurement and still needs integration into the verified snapshot.

The encode setup correction passes all 17 focused encode tests (1.11 seconds
runtime on MSI). An unavailable whole-chart derivative bound is now an explicit
capability absence: the atlas retains exact-solve centers and publishes no
amortized predictor or certificate. Precomputed, quotient, Möbius, finite-set,
and Duchon atoms follow the same rule. The unreachable cubic-Duchon bound and
its unused radial metadata were removed. This fixes setup classification, not
the separate SAE criterion/ARD inconsistency; the original pair still needs
remeasurement against the integrated build.

## Integrated encode and proper smoothing posterior

The r5 executable measured **24 passes, four failures, and two timeouts**.
All 30 identities were present and the before/after binary hashes matched.
Receipts are in `bench/measurements/issue_2668/issue2668-source-r5/`.

The smoothing-posterior support/prior changes clear the cubature-domain error.
The original irrelevant-covariate statistic is now measured: mean EDF 2.030541
(1.119791, 3.320913, 1.967723, 2.554531, 1.189746 across the five seeds), above
the required 1.0. Supported-signal EDF averages 8.636143. This is a real
shrinkage failure, not an integration exception. The concave shape fit also
fails during inference (P-IRLS nonconvergence after two iterations, reported
gradient 2.293496e-7), so the r4 all-shapes pass does not carry forward to r5.

Both SAE regressions pass atlas setup. The EFS test now exposes a one-row
encode batch rejected by its row-indexed affine fixture. The collapsing-axis
criterion refuses an unconverged mode, with raw KKT 1.391429e-2 versus tolerance
7.044921e-4. NB covariance/prediction and survival location-scale EDF still
exceed 60 seconds. No remaining contract is counted resolved.

The next custom-family experiment uses one coefficient-accuracy policy for
both value and derivative evaluations of the Laplace criterion. A nonlinear
coefficient-dependent determinant has first-order sensitivity to coefficient
error; coarse line-search modes therefore change the numerical objective
relative to its gradient. The experiment also removes the residual-only
override of the inner solver's convergence verdict, which ignored negative
curvature. A quartic with an analytic mode, Laplace value, and rho derivative
provides an independent focused accuracy check. That public-API integration
test passes in 0.02 seconds. Its unique mode is beta=1 at rho=0, with
`V=-9/8+log(5/2)/2` and `dV/drho=3/25`; both evaluation modes meet the analytic
value within 1e-9 and agree with each other within 1e-12. The receipt is
`bench/measurements/issue_2668/laml-accuracy-integration.log`.

The monolithic unit-test build was stopped after it spent over three minutes
compiling unrelated test families. Moving the oracle into a dedicated integration
target and using 16 codegen units for gam-custom-family reduced its build to
26.80 seconds against the warm cache. This is the iteration target for further
criterion-accuracy work. Full regression impact is still under measurement.

## Common LAML accuracy integration

The r6 executable again measured **24 passes, four failures, and two timeouts**,
with all 30 identities and an unchanged binary hash. Receipts are in
`bench/measurements/issue_2668/issue2668-source-r6/`. The common accuracy policy
is independently verified, but does not resolve the NB end-to-end stall.
The first two NB seeds still stop around cost 1028.111, with terminal gradient
norms 9.426e-4 and 4.209e-4, above their certification bounds. Later inner
solves alternate between a decrement convergence claim and a returned-mode
revocation: one sample has decrement 1.076e-17 and coefficient correction
1.948e-9 against 3.768e-11, with observed objective changes near 1e-11. The
returned-mode check adds a step-floor condition absent from the earlier
decrement check. Neither threshold has been relaxed to count a pass.

The SAE identity fixture now evaluates the global function `phi(t)=t` on any
batch. EFS gets past the previous row-shape exception and reaches its actual
criterion check: after 6400 granted iterations it rejects a non-idempotent
inner map (7.80 seconds). The free-decoder scaling orbit remains unresolved.
The original assertions are unchanged.

The concave inference failure is localized to a smoothing cubature point
`rho=[18.558,-1.938]`, after two P-IRLS iterations. Its face decrement is
2.931131e-12 against 1.467844e-13. The adjacent point
`rho=[18.558,2.689]` converges in three iterations. A separate four-second
diagnostic from r5 retains the trace in `shape-r5-diagnostic.log`; r6 enables
the regression's diagnostic logger directly. Inspection also found that the
post-polish decrement check still excludes active inequality faces, whereas
the in-loop check already uses their tangent-space decrement. Its relevance
to the measured failure needs a focused test before changing production.

## Terminal constrained-face certificate

A focused quadratic reproduces a separate terminal P-IRLS defect. Its free
coefficient fits two adjacent floating-point observations with equal weights;
their exact mean lies between representable values. A constrained coefficient
has a binding lower bound and a nonzero multiplier. After one damped step makes
resolvable progress, undamped final refinement reaches the nearest representable
minimum, but the accepted-step plateau check has never seen that final point.
The old final check excludes active faces and returns `MaxIterationsReached`.

The final check now uses the same face-restricted Newton decrement as the
in-loop check. The fixture's weight is derived so its squared decrement is
exactly epsilon/2 (1.110223e-16), below the existing threshold 1.332268e-15;
its tangent gradient remains 1.525879e-5. No tolerance was loosened. The
single-case baseline fails and the corrected integration test passes for
lower-bound rows, equivalent general linear rows, and an all-infinite box
(0.03 seconds). The normalized inequality system also supplies the terminal
geometry check, including lower-bound feasibility. The now-unused
`inequalities_are_all_inactive` helper was removed.

Receipts: `face-before.log`, `face-after.log`, and `face-warm-compile.log` under
`bench/measurements/issue_2668/`. The before receipt covers the lower-bound
case; the final after receipt also covers the two equivalent representations.
The full r7 regression build is pending; the original concave-inference
contract is not yet counted resolved.

`scripts/compile_warm_probe.py` compiles a diagnostic against an explicitly
selected existing Cargo library graph, resolving direct dependencies by their
fingerprints and recording source/library/binary hashes. It avoids rebuilding
the dependency graph when a small probe changes. Use it only while the target
directory is idle: a later attempted baseline compilation overlapped r7 and
correctly failed when Cargo removed the old library before linking. That
attempt produced no test result and is excluded from the proof above.

## Final r7 verification (2026-09-08)

The completed r7 inventory has **24 passes, four failures, and two 60-second
timeouts**. All 30 original test identities are present, and the executable
SHA-256 remained `33645bfe107f0a34a0f55a16ece9bf7c1750d284ffce1b1f5847674a3950b237`
throughout the run. Raw logs, inventory, and results are retained under
`bench/measurements/issue_2668/issue2668-source-r7/`.

The terminal-face correction passes its focused analytic quadratic test, but
does not resolve the original concave inference failure: that still reports
two iterations and gradient norm 2.293496e-7. The other failures are the two
SAE ARD criterion contracts and irrelevant-covariate double-penalty EDF. NB
dispersion covariance and survival location-scale EDF still exceed 60 seconds.
NB generated/predicted variance agreement passes in 36.50 seconds.

The determinant-conditioning idea remains an unverified hypothesis. No
dispersion determinant policy change is included: the inner cached determinant
and outer derivative convention need to be examined together before changing
either. No additional build or experiment is running from this task; the
runner terminated its timed-out process groups. Issue #2668 remains open.
