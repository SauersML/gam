# Issue 932: reopening audit, 2026-09-08

The issue is **not verified complete**. This audit preserves the requirements
from the previous reopenings, including the owner's July 26 list. A green
subset of the release matrix cannot discharge the other items.

Sources: [issue and original deployment plan](https://github.com/SauersML/gam/issues/932),
[latest request to review every closure](https://github.com/SauersML/gam/issues/932#issuecomment-5589023539),
[September 5 closure](https://github.com/SauersML/gam/issues/932#issuecomment-5554274027).

## Current counterexamples

1. The newer [release run 34262907256](https://github.com/SauersML/gam/actions/runs/34262907256)
   at `9cda85f8a01565e4cbc1de4ed9c32007f58a0072` still fails rigid BMS
   contracted third (**0.991894**, wins 0.07, resolution 0.0064) and full
   third (**0.996361**, wins 0.13, resolution 0.0141). Earlier green evidence
   therefore does not establish a durable strict speed advantage.
   [Release run 34250437216](https://github.com/SauersML/gam/actions/runs/34250437216)
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
| Timewiggle-q compose layer faster than optimized analytic reference | Current full-output hand parity and release measurement of the disputed compose layer | Original analytic-basis opponent now beaten in isolated MSI measurements, with an enforced source gate; integrated release and strongest-hand schedule still pending |
| GPU end-to-end regression (reported 0.69x at n=32768/r=20) | Transfer-inclusive current CPU/GPU comparison at that shape and relevant dispatch behavior, with utilization | Pending; CUDA compilation alone is insufficient |
| SAE non-softmax/IBP/JumpReLU strongest-hand comparison | Equivalent live prior and reconstruction semantics, all-channel parity and optimized hand timing | Current independent-gate hand test exists; opponent and mode coverage audit pending |
| Runtime-width SLS wiggle, orders two through four | Runtime-sized analytic hand opponent and complete channel parity/timing | Incomplete: current test measures allocation policy |
| Constrained Firth/Jeffreys root cause | Correction enabled on identifiable geometry and converged constrained binomial-wiggle/Matérn fit, plus affected-family regressions | Incomplete: disabling rationale remains in production |
| Large-scale flex end-to-end benchmark | A real converged fit selecting the intended branch, cold/warm cache attribution, comparable baseline and timing | Pending; per-row allocation test is insufficient; inspect current Criterion target's capped-fit semantics |
| Retired hand fourth-order oracle reduced to finiteness | Independent numerical agreement through fourth order on the live route | Pending; finite-only agreement is not sufficient |
| Block10 fourth-order FD convergence omissions | Every required entry covered by a converged independent witness or exact oracle | Fixed and verified on MSI: all four fixtures, no skipped matrix entries, exact-zero checks for zero directions; original error bounds retained |
| Loosened oracle tolerances and narrowed fixtures | Justified numerical error bounds, wider relevant fixtures, corruption sensitivity | Pending; inspect each affected oracle, not only the repaired rigid test |
| Removed hand-oracle coverage | Independent replacement for each still-live channel, not a comparison of one lowering with itself | Pending |
| M=32 complete canonical fourth-order coverage without stack overflow | Executed bounded-stack live-route/canonical test, explicit matrix coverage, no width refusal | Passed on MSI: four fixtures, every 32×32 third/fourth entry, canonical evaluation on an explicit 1 MiB stack |
| Moment-order, implicit-lift, heap and CUDA tile costs | Measurements covering the live changes, including common widths at and below 32 | Pending; isolated primitive wins cannot establish total path speed |
| GPU initialization error distinguishes memory headroom from missing runtime | Current typed failure propagation and behavior tests | Verified at the runtime policy boundary on MSI: zero-memory present runtime survives Auto/Required resolution; allocation fault retains its typed diagnosis |

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
1.024316. All package jobs in that run have now completed successfully. This
whole-run verdict predates the newly enforced BMS flex gate and moving-edge
repair. Attempts to connect to acn116 returned SSH status
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

The GPU runtime policy regression
`exhausted_memory_does_not_erase_a_present_runtime_932` passed on MSI along with
all six existing policy-resolution tests in the standard release profile.
It injects a present device with zero free memory and a zero memory budget,
checks that both Auto and Required preserve that runtime, and separately checks
that `CUDA_ERROR_OUT_OF_MEMORY` remains a `DriverCallFailed` diagnosis. This
tests the actual policy resolver; it does not claim a physical GPU allocation
or end-to-end throughput measurement. Log:
`.buildd/issue932-gpu-memory-contract.log`.

The existing MSI model test binary passed **70 issue-932 correctness witnesses**
in 4.52 seconds, with serial execution and measurement tests excluded. This
includes the M=32 bounded-stack test, rigid independent full-tensor algebra and
corruption controls, Gaussian coefficient-map witnesses, survival joint and
wiggle Hessians, implicit lifts, moving boundaries, and selected-GPU error
propagation. The rigid tensor's worst scaled error was **7.694e-16** against
its **7.105e-15** bound. Log: `.buildd/issue932-current-model-witnesses.log`.
These are the sources compiled for the model diagnostic above, not a claim
that all concurrent working-tree edits have been built. A source review also
found that the SLS higher-order FD oracle's maximum-error accumulation could
hide NaNs. Its next revision rejects non-finite analytic and stencil entries
and extends the original Gaussian/Gumbel event fixture to Logistic and mixed
censoring, while retaining the existing numerical bounds.
That expanded oracle passed on MSI: 18 distribution/censoring/direction cases,
all matrix entries checked; the largest scaled fourth error stayed below
7.3e-6 against the unchanged 1e-4 bound. This correctness-only build disabled
optimization for the model package while retaining the warmed release
dependencies; no speed conclusion is drawn from it. Log:
`.buildd/issue932-sls-fd-expanded.log`.

The historical Block10 test now lives under
`flex_production_fourth_contraction_matches_scalar_fd_witness`. Running the
current optimized binary confirmed that its skip is still active: for example,
`zero_warp_edge/alternating->mixed` entry `[3,3]` produced -0.16347 while the
coarse/fine FD estimates were -0.16359 and +12.167. The test printed a skip and
passed. This remains a counterexample to complete fourth-order verification;
changing the test name and removing the old hand producer did not resolve it.
Log: `.buildd/issue932-block10-before.log`.

The moving-edge primitive had a separate concrete defect: it formed derivatives
of `z^n exp(-q)` with `n/z`, replacing that ratio by zero at `z=0`. For example,
at `n=1`, `eta=0`, the second derivative of the sliver at zero is 1, whereas
that implementation returned 0. Near zero, the inverse powers also overflow.
The repaired primitive differentiates the monomial without division and uses
the product rule with the exponential derivative stack. The new independent
Hermite-polynomial witness checks every nested derivative channel through
fourth order for n=0..4, at zero, ±1e-120, and ordinary positive/negative edges.
It passed on MSI, along with four existing moment FD witnesses and all **71**
issue-932 model correctness tests. Logs: `.buildd/issue932-edge-polynomial.log`,
`.buildd/issue932-edge-moment-witnesses.log`, and
`.buildd/issue932-edge-model-witnesses.log`. The model package was unoptimized;
release performance evidence for this repair is still pending.

The Block10 discrepancy remains after the edge repair, with the same skipped
entries (`.buildd/issue932-block10-edge-polynomial.log`). The local candidate
turns those omissions into assertions and checks zero-direction outputs
against exact zero; resolving its numerical counterexample is still required.

Publication: `877b32ca5` contains the moving-edge repair and audit. Concurrent
shared-index staging also included changes to `gam-math/src/jet_tower.rs`,
`gam-solve/src/constrained_gaussian_reml.rs`, and
`gam-solve/src/gaussian_reml.rs` in that commit. The tests above validate the
moving-edge/model candidate on MSI, not those concurrently staged changes.
They were preserved; no history or worktree restoration was performed.

The next Block10 investigation isolated a quadrature defect for the very wide
finite cells created by small nonzero slopes. Integrating one non-affine
polynomial on `[3,10000]` gave M0 **0.00331604250946477457**, while partitioning
the identical integral gave **0.00331602882076341855**. The new interval-additivity
regression failed before the repair. The candidate reduces only integration
tails whose Gaussian-envelope bound is below half the smallest f64 subnormal;
the original cell and moving-boundary geometry remain intact. The bound covers
every absolute moment through the requested degree. CPU and CUDA source
emission share its radius calculation. With this reduction, all **89** kernel
tests passed in release, including all 33 moment slots of the new regression.
Logs: `.buildd/issue932-wide-cell-before.log` and
`.buildd/issue932-wide-cell-after.log`.

The full Block10 gate now **passes without skipped entries**, including the
original zero-warp fixture. Its finite checks and zero-direction assertions
also passed, with the original numerical bands unchanged. Thus the wide-cell
quadrature defect, rather than an intrinsically nonsmooth calibration root,
caused the historical discrepancy. The current model binary additionally
passed all 71 issue-932 correctness witnesses, both independent flex contracted
tower witnesses, and all five CUDA source-emission tests. Physical CUDA
execution and release performance remain pending. Logs:
`.buildd/issue932-block10-wide-domain-result.log`,
`.buildd/issue932-wide-model-witnesses.log`,
`.buildd/issue932-wide-flex-witnesses.log`, and
`.buildd/issue932-wide-cuda-source.log`.

For the model integration check, direct rustc reused the existing warm
dependency artifacts with optimization and LTO disabled, four pinned CPUs,
and a 150-second build cap. It completed in 129.93 seconds. The exact compiler
arguments are saved in `.buildd/issue932-wide-direct-argv.json`. Earlier Cargo
attempts rebuilt dependency variants and hit development caps; those attempts
are not passing evidence. No local build or numerical execution was used.

The follow-up non-affine convergence audit found that `f64::max` silently
discarded NaN moments before the reduced maximum-error finite check. A corrupted
coarse moment could therefore certify convergence. The new corruption witness
failed on the original implementation at moment zero, then passed after adding
per-moment finite checks. It checks NaN and both infinities in each of the 33
coarse/fine slots, including matching corrupted pairs, while retaining positive
controls for finite and exact-zero agreement. All **90** kernel tests passed on
MSI after the repair. The warm direct-rustc build took 9.64 seconds at opt-level
2 with four pinned CPUs; tests took 0.15 seconds on one CPU. Logs:
`.buildd/issue932-ladder-before.log` and `.buildd/issue932-ladder-after.log`.
These are correctness results; no speed conclusion is drawn from this build.

The timewiggle coefficient map now expresses a production invariant in its
scalar interface: coefficients are fixed in the outer family direction, while
their inner coefficient jets remain live. This removes construction and
multiplication of zero family-derivative jets without dropping mixed
coefficient/family channels. Exit value and slope are evaluated together, with
identical supplied polynomial stacks sharing their complete weighted term.
The existing family-geometry body is unchanged; the scalar program and its
tests moved to `timewiggle_geometry/scalar_q.rs`. A small MSI harness includes
that production source directly instead of copying its implementation.

The new paired gate first failed on the original arithmetic: analytic/production
ratios **0.846274 / 0.864490** for `Dual2<Order2<5>>` and
`Dual2<OneSeed<5>>`, both wins 0.00. The final candidate passes at
**1.111911 / 1.211732**, both wins 1.00, resolutions 0.0047 / 0.0117. Each
measurement uses 15 interleaved repetitions and consumes every channel from 64
varied rows per arm call. The opponent is the historical polynomial/exp scalar
program, **not** a fully hand-expanded runtime-width spline schedule. This does
not discharge that separate requirement. The benchmark uses opt-level 3, one
codegen unit and LTO off; integrated release CI remains required. Logs:
`.buildd/issue932-timewiggle-baseline.log` and
`.buildd/issue932-timewiggle-final.log`.

All five isolated tests pass, including a new direct-polynomial witness at
widths 0, 1, 2, 7 and 32 with nonzero coefficient Hessians and mixed family
channels. The MSI model integration build completed in **129.40 seconds** with
the existing warm dependencies, optimization disabled, four pinned CPUs and a
150-second cap. It passed all **72** issue-932 correctness witnesses (8.25 s)
and all **28** timewiggle integration tests (31.92 s). The latter include the
public family/design workspace, joint-Hessian FD, operator/dense agreement,
and baseline-family derivative consumers. Logs:
`.buildd/issue932-timewiggle-model-build.log`,
`.buildd/issue932-timewiggle-model-witnesses.log`, and
`.buildd/issue932-timewiggle-integration.log`. CI's marker parser finds
`release_timewiggle_q_vs_analytic_basis_program_932`, and the integrated binary
lists exactly that new test.

The completed model job `102185652642` in release run **34262907256** also
validates the shortened BMS flex gate: all four cells pass, with dynamic/fixed
ratios **1.049541 / 2.866422** for score-warp orders 3/4 and
**1.021672 / 4.639822** for link-dev orders 3/4; all wins are 1.00. Its opponent
remains a dynamic jet. The run as a whole fails the two rigid macro cells
listed above, so it is not a whole-population acceptance result.
