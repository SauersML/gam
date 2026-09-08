# Issue 932: reopening audit, 2026-09-08

The issue is **not verified complete**. This audit preserves the requirements
from the previous reopenings, including the owner's July 26 list. A green
subset of the release matrix cannot discharge the other items.

Sources: [issue and original deployment plan](https://github.com/SauersML/gam/issues/932),
[latest request to review every closure](https://github.com/SauersML/gam/issues/932#issuecomment-5589023539),
[September 5 closure](https://github.com/SauersML/gam/issues/932#issuecomment-5554274027).

## Current counterexamples

1. [Release run 34250437216](https://github.com/SauersML/gam/actions/runs/34250437216)
   at `d82efb558d2119c784a867ecd9378062e8ff83ae` failed two macro speed cells:
   rigid BMS `third_full`, hand/generated median ratio **0.932480**, wins
   **0.00**, resolution **0.0057**; SLS V/G/H **0.992762**, wins **0.07**,
   resolution **0.0061**. These are executed release failures, not skipped
   tests. A prior green run does not resolve these observations.
2. `survival/location_scale/row_kernel.rs` explicitly describes
   `SLS-WIGGLE-DYN-932` as an arena-amortization diagnostic, not a
   strongest-hand comparison. Orders two and three compare the same jet
   evaluator with reused versus fresh storage. Order four checks retained
   arena capacity without a timing. This does not satisfy the July reopening's
   demand for a manually optimized derivative opponent at runtime width.
3. `gamlss/binomial/wiggle_custom_family.rs` still documents disabling the
   Firth/Jeffreys term to avoid the structural-null curvature wall. The July
   reopening explicitly requested fixing the underlying constrained fit.
   Information derivative hooks alone do not prove that the correction is
   enabled or that such fits converge.
4. `bms/flex_measure_932_tests.rs` retained a separate best-of-five timer,
   compared only two matrix entries during timing, and asserted no speed
   verdict. It therefore escaped the `SpeedGate::open`-derived release
   population. The current candidate replaces that timer with paired,
   interleaved measurements consuming every matrix entry and opens the gate
   in the test body. Its opponent remains explicitly a **dynamic jet**;
   this repairs enforcement of specialization performance, not the missing
   strongest-hand comparison.

## Requirements and evidence still needed

| Requirement from issue history | Authoritative completion evidence | Audit status |
| --- | --- | --- |
| One row likelihood definition for every live family, including coefficient maps and mixed blocks | Enumerate live consumers and follow each V/G/H/T3/T4 implementation to the canonical expression; no independent production derivative authority | Re-audit required; macro presence alone is insufficient |
| Stable primitive algebra through fourth order | Independent primitive witnesses, including tails, activity, and overflow cases; executed relevant tests | Prior evidence exists; current validation pending |
| All-channel universal oracle in CI | Independent values and all derivative channels, non-finite rejection and corruption controls, executed in current source | Rigid oracle repair is present; whole-population validation pending |
| Survival location-scale joint Hessian and directional cutover | Live joint-Hessian capability and callers, all three residual distributions and entry/exit/event channels | Current source and runtime audit pending |
| Implicit roots and moving cell boundaries | One implemented value/derivative expression, differentiated maps, and independent boundary/implicit witnesses | Current source and runtime audit pending |
| Strongest-hand speed evidence for every shipped lowering | Same-contract, same-input, same-output, paired release measurements; full matrices consumed | Incomplete: runtime SLS and BMS flex proxy opponents remain; macro failures above |
| Continuous release enforcement | Derived tests resolve and execute; no unasserted timer or skipped cell counted as acceptance | BMS flex enforcement repair under validation; runner coverage audit pending |
| Timewiggle-q compose layer faster than optimized analytic reference | Current full-output hand parity and release measurement of the disputed compose layer | Pending; primitive or arena timing is not sufficient |
| GPU end-to-end regression (reported 0.69x at n=32768/r=20) | Transfer-inclusive current CPU/GPU comparison at that shape and relevant dispatch behavior, with utilization | Pending; CUDA compilation alone is insufficient |
| SAE non-softmax/IBP/JumpReLU strongest-hand comparison | Equivalent live prior and reconstruction semantics, all-channel parity and optimized hand timing | Current independent-gate hand test exists; opponent and mode coverage audit pending |
| Runtime-width SLS wiggle, orders two through four | Runtime-sized analytic hand opponent and complete channel parity/timing | Incomplete: current test measures allocation policy |
| Constrained Firth/Jeffreys root cause | Correction enabled on identifiable geometry and converged constrained binomial-wiggle/Matérn fit, plus affected-family regressions | Incomplete: disabling rationale remains in production |
| Large-scale flex end-to-end benchmark | A real converged fit selecting the intended branch, cold/warm cache attribution, comparable baseline and timing | Pending; per-row allocation test is insufficient; inspect current Criterion target's capped-fit semantics |
| Retired hand fourth-order oracle reduced to finiteness | Independent numerical agreement through fourth order on the live route | Pending; finite-only agreement is not sufficient |
| Block10 fourth-order FD convergence omissions | Every required entry covered by a converged independent witness or exact oracle | Pending; skipping unresolved entries cannot prove all-channel correctness |
| Loosened oracle tolerances and narrowed fixtures | Justified numerical error bounds, wider relevant fixtures, corruption sensitivity | Pending; inspect each affected oracle, not only the repaired rigid test |
| Removed hand-oracle coverage | Independent replacement for each still-live channel, not a comparison of one lowering with itself | Pending |
| M=32 complete canonical fourth-order coverage without stack overflow | Executed bounded-stack live-route/canonical test, explicit matrix coverage, no width refusal | Current source and runtime audit pending |
| Moment-order, implicit-lift, heap and CUDA tile costs | Measurements covering the live changes, including common widths at and below 32 | Pending; isolated primitive wins cannot establish total path speed |
| GPU initialization error distinguishes memory headroom from missing runtime | Current typed failure propagation and behavior tests | Current source and runtime audit pending |

## Validation record

Commit `c9c13b66f7c972e12046331b0a431b2ee3bcf60b` fixes redundant
full-third tensor emission. The root composition now computes its four unique
two-primary components once and copies them to the symmetric tensor slots.
The new permutation regression failed before the change (separately evaluated
permutations differed in floating-point bits), then passed after it. Independent
hand-formula agreement remains checked over all 512 rows.

MSI acn112 release validation passed all 27 macro unit tests and all 10 macro
integration tests. The analytic full-third opponent was then strengthened to
compute its four symmetric components once too, with explicit multiplicities.
Against that stronger opponent, the focused rigid suite passed both tests and
all five speed cells: full-third hand/generated **1.019530**, wins **1.00**,
resolution **0.0025**, 15 paired repetitions. Logs:
`.buildd/issue932-third-symmetry-before.log`,
`.buildd/issue932-third-symmetry-after.log`, and
`.buildd/issue932-third-symmetric-hand.log`.

Publication is verified: remote main commit
`43535bd843ee3617f9d6f1a5e27b0c8454258110` includes `c9c13b66f` as a parent;
both changed blobs match the local commit. Its release CI run
[34257608661](https://github.com/SauersML/gam/actions/runs/34257608661)
has completed its macro job successfully (job `102167379883`, EPYC 7763):
all five derived gates passed. Full-third measured **1.060940**, wins **1.00**,
resolution **0.0047**, against the strengthened hand opponent. The other rigid
cells measured 1.024296 / 1.145698 / 1.178569 / 1.343110 for order two,
contracted third, contracted fourth, and full fourth. SLS macro V/G/H measured
1.024316. The remaining package jobs are not yet a completed whole-population
verdict. Attempts to connect to acn116 returned SSH status
255 before starting the test; there is no second-host measurement to report.

The BMS flex candidate is being validated on MSI in the existing
`/projects/standard/hsiehph/sauer354/gam-main-validation` source directory using
the warm `/scratch.global/sauer354/y5-target` cache, four Cargo workers and
serial tests. The first attempt inherited invalid OpenBLAS linker flags;
clearing them recovered the warm cache. The one-codegen-unit test-crate build
was stopped after several minutes to shorten development iteration. The active
candidate check used the release profile with the model package set to 16
codegen units; it passed, but is **not** final production-profile speed evidence. Log:
`.buildd/issue932-flex-paired-cgu16-20260908.log`.
All four width-12 cells passed, each with wins 1.00: score-warp third/fourth
dynamic/fixed ratios **1.138264 / 3.215181** and link-dev third/fourth
**3.014237 / 3.994068**. The initial 128-iteration timing took 96.25 seconds;
the candidate now uses 16 iterations per sample, retaining all 15 paired
repetitions. Even its fastest observed arm then contributes about 14 ms per
sample. The shorter measurement still needs the integrated release CI verdict.
The CI discovery function has separately confirmed that it derives the newly
wired `release_measure_bms_empirical_third_fourth_fixed_vs_dynamic_932` test.
No local compilation or test execution is used. This validation checks the
candidate benchmark change; it does not certify unrelated uncommitted files
or discharge the rest of this table.
