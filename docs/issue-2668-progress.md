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
