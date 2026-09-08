# Issue 2668: original-contract verification

The acceptance scope is every one of the issue's original 30 entries, including
the two deleted contracts. Moving failures to another issue is not completion.
The issue remains unresolved.

`tests/data/issue_2668_regressions.json` records all 30 original identities. It
explicitly maps the two previously documented replacements. The runner resolves
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
