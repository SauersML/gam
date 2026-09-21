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
| One row likelihood definition for every live family, including coefficient maps and mixed blocks | Enumerate live consumers and follow each V/G/H/T3/T4 implementation to the canonical expression; no independent production derivative authority | Source census 09-12 (the two live-family derivative census sections below): every live family's joint-Hessian derivative traces to one row expression, to a 1-D primitive, or to a hand closed form pinned against the expression. Its last residual was the BMS flex coefficient-space `D_β H[v]` and `D²_β H[u, v]`, covered only by identity tests. `standard_normal_flex_beta_hessian_directional_derivatives_match_finite_difference_932` (`9b0a73a6c`) checks both against Richardson differences of the dense Hessian over six rows with two-column designs. It failed in pool job 607307 at `004fa46a6` (the marginal/slope × link-deviation block of `D_β H[v]` up to 7.6 bars, `D²_β H` up to 30.8), before spec-derivs' link-crossing terms reached the t3 and V/G/H lowerings (`2d79c690d`, `62ff43758`, `882e6ce2e`). It passed in pool job 617386 at `04da1e5d7` (acn12, EPYC 7763): `D_β H[v]` worst 2.6e-3 of its bar, `D²_β H` 6.2e-3, differences resolved to 2e-7 of scale |
| Stable primitive algebra through fourth order | Independent primitive witnesses, including tails, activity, and overflow cases; executed relevant tests | Source audit 09-12, normal log-CDF stack (`gam-math/src/probability.rs` tests), the leaf every BMS and SLS tower composes on: mpmath 100-digit references for the fifth and sixth derivatives at x = −100, −20, −8, −4, −2, 0, 2, 8, 20 (2e-10 relative); 400/450-digit references at x = 38.6 (4 subnormal ulps); central differences for orders 1–3 at x = −8, −4, 8, 20 (2e-5) and fifth/sixth vs the order below at ten points from −100 to 20 (3e-5); exact infinite and NaN limits, a non-cancelling −1e100 left tail, a subnormal-preserving 38.6 right tail, and bit-identical `_through_fifth`/`_through_sixth` across every branch edge. Orders 1–4 have difference and exact-tail witnesses, no high-precision table. `signed_probit_stack_preserves_extreme_tail_derivatives_and_weight_sign` pins the signed stack's infinite-margin, weight-sign and tail semantics exactly. Residual distributions and inverse links carry FD witnesses (`residual_pdfthird_derivative_matchessecond_derivativefd`, `residual_pdffourth_derivative_matches_independent_fd_witness`, `inverse_link_pdfthird_derivative_matches_d3_finite_difference`); the sqrt and power stacks are pinned to their closed forms by `sqrt_and_power_derivative_stacks_match_closed_forms_932` (`c74fbc04b`, passed in pool job 559612), and the gamma stack is not enumerated. Not run in the 09-12 pass |
| All-channel universal oracle in CI | Independent values and all derivative channels, non-finite rejection and corruption controls, executed in current source | Rigid oracle repair is present; whole-population validation pending |
| Survival location-scale joint Hessian and directional cutover | Live joint-Hessian capability and callers, all three residual distributions and entry/exit/event channels | Source audit 09-12: every `SurvivalLsRowKernel` surface (order 2, third, fourth, and fifth through `row_nll_fifth_inputs_opt`) is one `sls_row_program` declaration. Witnesses on main, over Gaussian/Gumbel/Logistic: `survival_ls_time_varying_joint_hessian_tower_body` (tower H vs packed assembler, 1e-9; deaths, right censoring, deep left truncation, extreme exit tails, fractional event); `survival_ls_block_gradient_tower_body` (1e-9); `survival_ls_joint_directional_derivative_time_varying_body` (hand `_rescaled_from_parts` vs tower-certified generic directional, 1e-7, six directions); `survival_ls_third_directional_all_axes_matches_difference_of_second_2677` (1e-6); `sls_index_sparse_lowering_matches_independent_fd_all_branches_932` (high-order FD at d = 0, 1, 0.37; 2e-7 / 2e-5). Not run in the 09-12 pass |
| Implicit roots and moving cell boundaries | One implemented value/derivative expression, differentiated maps, and independent boundary/implicit witnesses | Source audit 09-12: `filtered_implicit_solve_scalar` is one `JetScalar`-generic implicit solve, used by the empirical-rigid intercept jet (`bms/cell_moment_assembly.rs`); gam-math tests pin linear (1e-14) and quadratic analytic-IFT (1e-12) constraints, and the empirical-rigid polynomial oracle covers the live route (434 entries, 1e-9). `implicit_intercept_fifth_composition` feeds `empirical_rigid_row_fifth_full`, which is pinned by `empirical_fifth_tensor_matches_derivative_of_fourth_979` (central differences of the production fourth tensor, 2e-7) and by the degree-5 channels of the independent polynomial oracle `empirical_rigid_all_channels_match_independent_polynomial_932` (all 32 entries, 1e-9). `moving_edge_leibniz_tracks_boundary_flux_932` checks value 1e-9, first 1e-6, second only 1e-3 against a 6000-panel quadrature, so the second-order bar stays open. Not run in the 09-12 pass |
| Strongest-hand speed evidence for every shipped lowering | Same-contract, same-input, same-output, paired release measurements; full matrices consumed | Incomplete: runtime SLS and BMS flex proxy opponents remain; macro failures above |
| Continuous release enforcement | Derived tests resolve and execute; no unasserted timer or skipped cell counted as acceptance | BMS flex enforcement repair under validation; runner coverage audit pending |
| Timewiggle-q compose layer faster than optimized analytic reference | Current full-output hand parity and release measurement of the disputed compose layer | Original analytic-basis opponent now beaten in isolated MSI measurements, with an enforced source gate; integrated release and strongest-hand schedule still pending |
| GPU end-to-end regression (reported 0.69x at n=32768/r=20) | Transfer-inclusive current CPU/GPU comparison at that shape and relevant dispatch behavior, with utilization | Pending; CUDA compilation alone is insufficient |
| SAE non-softmax/IBP/JumpReLU strongest-hand comparison | Equivalent live prior and reconstruction semantics, all-channel parity and optimized hand timing | Source audit 09-12: production `AssignmentMode` is {Softmax, OrderedBetaBernoulli, ThresholdGate, TopK}; no JumpReLU mode exists. `row_jets_for_logdet` routes Softmax through `execute_softmax_row_program`, OrderedBetaBernoulli and ThresholdGate through the shared `execute_independent_logistic_row_program`, and TopK through that schedule's constant-gate degeneration (no logit primaries). `SAE-SOFTMAX-SCHEDULE-932` and `SAE-INDEPENDENT-SCHEDULE-932` race each compiled schedule against `row_jets_for_logdet_hand_reference`, the historical non-abstracted hand assembly, at K = 1, 2, 8, 16, 32, 64 and P = 16. Each first requires full-channel parity at 2e-12 of scale and no more allocations than the hand (exactly 2). The independent fixture is OrderedBetaBernoulli. ThresholdGate shares its program and still has no speed cell of its own, but `threshold_gate_compiled_schedule_matches_hand_full_channels_932` (`e0f3b5402`) pins its full-channel parity and allocations at every width. Not run in the 09-12 pass |
| Runtime-width SLS wiggle, orders two through four | Runtime-sized analytic hand opponent and complete channel parity/timing | Stage 1 of 3. `sls_wiggle_hand_932_tests.rs` (`04da1e5d7`) holds a hand schedule for the per-row `KW × KW` Hessian of `sls_row_nll_wiggle`: three rank-one terms plus the sparse intermediate Hessians. It is checked against production's `DynamicOrder2` lowering at widths 3 and 7 on event, censored and untruncated rows. It passed in pool job 617386 at `04da1e5d7`: worst entry 5.4e-5 of the `1e-11·max(1, \|a\|, \|b\|)` band, and a one-ppm `dr1` corruption trips the band 8.1e4-fold. The release cell `SLS-WIGGLE-HAND-932` (production not_slower than the hand at widths 3 and 12) has not run: that lane had no release target. The hand third and fourth contractions are not written. `SLS-WIGGLE-DYN-932` still times the arena, not the derivative schedule |
| Constrained Firth/Jeffreys root cause | Correction enabled on identifiable geometry and converged constrained binomial-wiggle/Matérn fit, plus affected-family regressions | Not started in code. Lead ruling 09-12: `BinomialLocationScaleWiggleFamily` and `BinomialLocationScaleFamily` arm through jeffreys2898's shared unarmed-first lifecycle (#2898 slice 2), with no second lifecycle and no one-observation cut. That withdraws the measured-span route this row proposed on 09-12. Source reading 09-13: the per-block route builds the full span (`jeffreys.rs:113-116` passes a zero aggregate), and the full span contains each family's likelihood gauge. When the log-σ design carries an intercept, `BinomialLocationScaleFamily`'s `q = −η_t/σ` is invariant along `(δη_t = η_t, δη_ls = 1)` (`location_scale.rs:2051`), so the information is singular along that direction at every β, not only at β = 0. In the wiggle family the warp's linear element rescales `q₀` (the #2647 budget test's doc). `ker(S)` excludes both gauges: `builders.rs:576` appends an identity shrinkage to the whole log-σ block, and the #2647 closure penalizes the linear warp (`H_L + S` λ_min 0.725 at β = 0, `tests_2647_gauge.rs:155`). But `jeffreys_subspace.rs:1336` records that `ker(S)` cannot reach separation along a penalized direction. A span taken from the typed arming evidence (the descending ray or the prefit separation direction) holds no gauge direction. Which span slice 2's armed refit uses was asked of jeffreys2898 on 09-13; the family side waits on that answer |
| Large-scale flex end-to-end benchmark | A real converged fit selecting the intended branch, cold/warm cache attribution, comparable baseline and timing | Pending; per-row allocation test is insufficient; inspect current Criterion target's capped-fit semantics |
| Retired hand fourth-order oracle reduced to finiteness | Independent numerical agreement through fourth order on the live route | BMS FLEX route covered on main by `standard_normal_flex_canonical_derivative_ladder_matches_vgh_t3_t4_932`: central differences through the production `lower_bms_flex_row_order2_with_moments` / `row_primary_third_contracted_with_moments` / `row_primary_fourth_contracted_ordered` along one mixed direction; V→G 2e-7, G→H 2e-6, H→t3 2e-5, t3→t4 2e-4; exact symmetry and nonzero-signal asserts. Not run in the 09-12 pass. Survival marginal-slope flex route (source reading 09-13): the flex third and fourth contractions over `(q0, q1, qd1, g)` match the independent rigid `Tower4` program at zero deviation, with a planted sign-flip control; `flex_production_fourth_contraction_matches_scalar_fd_witness` differences the third along g/h/w; `survival_flex_fourth_contraction_matches_differenced_third_along_q_axes_932` (`9829ccdbb`) differences it along a coefficient direction whose primary image moves q0, q1 and qd1, and passed in pool job 612572 at that sha (acn66, EPYC 7763). The BMS flex coefficient-space surfaces over six rows are row 48's |
| Block10 fourth-order FD convergence omissions | Every required entry covered by a converged independent witness or exact oracle | Fixed and verified on MSI: all four fixtures, no skipped matrix entries, exact-zero checks for zero directions; original error bounds retained |
| Loosened oracle tolerances and narrowed fixtures | Justified numerical error bounds, wider relevant fixtures, corruption sensitivity | Item status 09-12. (1) Empirical-rigid implicit solve, loosened 1e-9 → 1e-8 at T3/T4: back to 1e-9 through the independent polynomial oracle, now degree five (`a4fbf57f4`, passed in pool job 578401). (2) Rigid Bernoulli hand-chain witness: the band was `1e-12 + 1e-9·max(|a|, |b|)`. Pool job 603819 at `7427b7e91` measured the worst of 98 entries: value bit-identical, gradient 8.3e-16 and Hessian 6.8e-16 relative (3.75ε). `004fa46a6` holds the witness to a derived 64ε relative band and adds a one-part-in-10¹² corruption of `q1` that must trip every row. Pool job 607307 at `004fa46a6` passed it, weakest trip 70 times the band. (3) SAE cache-seam oracle with the mixed 1e-12 floor: deleted by `c0a21b554` with no census row while two doc comments still cite it; handed to restore2818. (4) `base_moment_jets` percent-level difference bars: replaced by an exact θ-Taylor oracle through order five on four cells (`3dc5240be`). It passed in pool job 584112 at `c7823aadd`: worst error 4.1e-5 of the `1e-10·(1 + |oracle|)` bar, and a one-ppm velocity corruption exceeds the bar 46–3877-fold at every derivative order from one to five (order zero reads no velocity). (5) BMS flex deviation ramp narrowed to 0.06: score-warp and link-dev arms at 0.25 added (`4d504767d`). Both passed in job 584112, score-warp worst Hessian at 1.59e-4 of its bar, link-dev max `|H − FD|` 1.78e-9 |
| Removed hand-oracle coverage | Independent replacement for each still-live channel, not a comparison of one lowering with itself | Source reading 09-13, survival marginal-slope, whose cleanup deleted 11,264 lines, including four historical hand timepoint-oracle files. On main, `flex_contracted_tower_matches_independent_rigid_tower_and_catches_sign_flip` holds the flex third and fourth contractions over `(q0, q1, qd1, g)` at zero deviation to the independent rigid `Tower4` program; a planted sign flip must leave the band. `flex_production_fourth_contraction_matches_scalar_fd_witness` checks the production fourth contraction against a Richardson difference of the production third along g/h/w directions, re-solving the intercept at every point. At nonzero deviation, `survival_flex_fourth_contraction_matches_differenced_third_along_q_axes_932` (`9829ccdbb`) checks the production fourth contraction along a coefficient direction whose primary image moves q0, q1 and qd1 against a Ridders difference of the third, and passed in pool job 612572 at that sha. The coefficient pullbacks have difference gates (`flex_no_wiggle_beta_hessian_directional_derivatives_match_finite_difference_932` and the #2893 time-wiggle gate), and the base-moment jets have an exact oracle (row 63 item 4). The two older witnesses named above were not run in this pass. The third-contraction difference witness at nonzero deviation is cited from a doc comment, not read. Other families' deleted hand oracles are not yet enumerated |
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
`$MSI_HOME/gam-main-validation` source directory using
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

The next rigid-third compiler change shares one root Faà di Bruno component
emitter between full and contracted surfaces. It combines identical partitions
with their integer multiplicity and emits each symmetric component once. The
contracted hand opponent now also uses three unique entries and folds its
diagonal multiplicity; the 512-row symmetry witness covers both surfaces.

A fresh MSI `acn112` release build took **12.31 seconds** and passed all **27
unit tests and 10 integration tests**. Hand/generated paired ratios were
**1.019350 / 1.074325 / 1.147608 / 1.060885 / 1.318502** for order 2,
contracted third, contracted fourth, full third and full fourth respectively;
all five cells had wins 1.00. Third/full-third resolutions were 0.0011/0.0031.
The run used the existing warm Cargo cache, four CPUs (88–91), 15 paired
repetitions and 512 rows per arm. Log:
`.buildd/issue932-third-partitions-verified.log`.

This run explicitly rebuilt the crate, and the suite binary timestamp
(14:20:02) follows both final source timestamps (14:17:15–16). The earlier
`issue932-third-partitions-strong-hand.log` used a stale binary and is excluded
from final-candidate evidence. Validation used the shared MSI checkout, which
also contains concurrent macro work; clean committed-source CI remains
required. A second-host attempt on `acn116` failed before execution (status
255). These results repair the local rigid-third gates, not the outstanding
whole-population, runtime-width hand-opponent, or physical-GPU requirements.

The next source audit found that the empirical-rigid module retained a comment
describing an exact fourth-order oracle after that oracle had been removed.
Its remaining independent finite-difference test allows percent-level errors
in the highest channels. The replacement uses a separate 15-coefficient
bivariate Taylor polynomial, ordinary coefficient convolution, and a
degree-by-degree solution of the calibration identity. Its CDF uses `libm`
and its log composition is independent of the production signed-log-CDF
derivative stack. It checks all 434 ordered V/G/H/T3/T4 entries over seven
row regimes, both frailty settings, and an exact-zero slope, at a fixed
**1e-9** scaled bound. Corruption controls reject non-finite pairs, sign
flips, and a one-sided 1e-8 disagreement.

Historical commit `84678107c` identifies the finite-only retired `cpu_oracle`
as the secondary arm of the same Block10 test repaired above. The live test
now numerically checks every matrix entry in that original g/h/w directional
scope without skipping unconverged stencils; the defective retired hand arm
is gone. This establishes the replacement's provenance, without claiming
coverage of additional directions from that test alone.

MSI verification passed all five empirical-rigid tests, including the new
polynomial oracle and corruption controls, in **0.13 seconds**. Maximum scaled
errors by order were **8.261e-16 / 7.043e-15 / 4.596e-14 / 3.426e-13 /
1.597e-12**. The model build took **129.39 seconds**, reusing warm dependencies
with optimization/LTO disabled and four pinned CPUs; these are correctness
results, not speed evidence. Logs: `.buildd/issue932-empirical-model-build-aligned.log`
and `.buildd/issue932-empirical-polynomial.log`; compiler arguments:
`.buildd/issue932-empirical-model-argv.json`. An initial attempt combined a
committed source snapshot with concurrent fifth-order callers and failed to
compile; the successful run uses the consistent shared MSI checkout. Clean
committed-source CI remains pending. No local compilation or numerical
execution was used.

## 2026-09-12 swarm pass (jet932)

Under the 09-12 swarm roster, jet932 holds all of #932 and jet932b holds #2333.
This section records the #932 landings `c38faa742` and `11a52ed5f`, recovered
from the previous jet932 session's unlanded patches, and three source censuses.

### BMS FLEX specialization cell, red on EPYC 9V74

Speed Gates run 34661849521 at `b66a7d04e` (gam-models job 103472091878, AMD EPYC
9V74) failed `BMS-FLEX-CONTRACTED-932` at order 3. The production_fixed/dynamic_jet
median ratio was **0.939753** for score-warp (wins 0.07, resolution 0.0178) and
**0.943020** for link-dev (wins 0.00, resolution 0.0053). Order 4 passed at
2.865857 and 4.640253. Run 34653246193 at `6c1cfb4ae` (job 103448845622) ran on the
same CPU model and passed order 3 at 1.065264 and 1.021045, both with wins 1.00.
The fixed-width third specialization therefore had no margin.

Cause, by reading `crates/gam-math/src/jet_scalar.rs`: `OneSeed<K>`'s `JetScalar`
impl overrode only `constant` and `variable`. `FixedRuntimeJet<OneSeed<K>, K>`
forwarded `linear_combination`, `multiply_add`, `composed_sum`, `affine_compose`
and `affine_composed_sum` to the trait defaults, which are chains of complete
`2·(1+K+K²)`-channel temporaries. It also left `weighted_compose_sum` to the
`RuntimeJetScalar` default loop, which does one composition and one product per
coefficient, per calibration node, per lift iteration. The opponent,
`DynamicOneSeedBatch`, already overrides `linear_combination`, `multiply_add`,
`affine_composed_sum` and `weighted_compose_sum`.

Change `c38faa742`: `OneSeed<K>` lowers those operations directly. `JetScalar`
gains `weighted_compose_sum`, whose default is the same loop. `Order2` overrides
it, and `FixedRuntimeJet` forwards it. `add_constant` keeps its default, so the
`row_program!` lowerings that the SLS SIMD bit-identity tests compare against do
not move. `one_seed_fused_932_tests` pins every override against its unfused field
program at 1e-13 relative, on operands whose ε Hessian channels are all live.

Receipts: pool job 440629 ran at `5d498d0e9` plus the patch. It passed the ban
scanner, 254 gam-math lib tests (including the three fused tests), the three
`flex_measure_932_tests`, 29 + 11 gam-row-macros tests, and
`cargo check --workspace --exclude gam-pyffi --all-targets`. Its mutation arm did
not compile, so no positive control ran. The release speed verdict comes from the
first Speed Gates run that contains `c38faa742` and completes, not from a lane run.

### GLM and exponential families: row derivative census

The observed-information tower for the GLM families is derived mechanically in
`crates/gam-solve/src/pirls/curvature.rs`. `weight_ratio_tower` forms
`T = h₁/(φV)` and its first four η-derivatives in one Leibniz recurrence on
`T·φV = h₁`. Its inputs are the inverse-link jet `h₁..h₅` (`mixture_link.rs`) and
the variance jet `V..V₄`. `W`, `∂W/∂η`, `∂²W/∂η²` and `e_obs = ∂³W/∂η³` are
polynomials in that tower and the jet. These 1-D primitives are the ones this
issue expects to keep. Beside the recurrence, `observed_weight_dispatch` carried
five closed-form specializations:

- **Gaussian log, Gaussian inverse and Binomial logit** were never reached in
  production. The dispatch has one production caller,
  `compute_observed_hessian_curvature_arrays_into`. It asserts
  `supports_observed_hessian_curvature_for_likelihood`, which admits only Gamma
  (any link), negative binomial under the log link, and the non-canonical Binomial
  links. `11a52ed5f` deletes these three arms and their helpers, reduces
  `WeightLink` to {Log, Other}, and drops the dispatch's `eta` parameter, which
  only the Gaussian-inverse arm read.
- **Gamma log and NB2 log** are live. Each closed form avoids the generic tower's
  large-η intermediates, and
  `gamma_log_observed_curvature_dispatch_avoids_generic_overflow` pins the Gamma
  case. `11a52ed5f` pins both `(W, ∂W/∂η, ∂²W/∂η²)` triples against Dual3
  derivatives of `−∂²ℓ/∂η²`, written from each log-likelihood
  (`gamma_log_observed_curvature_matches_dual3_932`,
  `negative_binomial_log_observed_curvature_matches_dual3_932`).
- **`e_obs`** has one production consumer, `hessian_cde_arrays`. It is reached
  only from `tierney_kadane_terms`, directly and through `hessian_cdef_arrays`.
  That correction returns zero unless `reml_robust_jeffreys_link` resolves: Firth
  bias reduction must be requested on a Binomial link with a Fisher-weight jet. No
  Gamma or negative-binomial fit therefore reads a third derivative. The other
  `e_obs_from_jets` callers sit in `reml/objective.rs`'s `tk_math_tests`.
  `11a52ed5f` corrects the `hessian_cde_arrays` prose that listed GammaLog.

Pool job 530387 runs the new pins and the #2273 dispatch tests on `11a52ed5f`.

### BMS oracle census: which bars guard production channels

The #932 tests in `crates/gam-models/src/bms` fall into two groups, and they
answer the loosened-tolerance row differently.

- **Percent-level finite-difference bars** gate the `#[cfg(test)]` jet substrate
  in `bms/test_support.rs`, not a production lowering:
  `cell_base_moment_jets_match_fd_932` (Hessian 1e-3 relative),
  `cell_base_moment_jets_moving_match_fd_932` (2e-3),
  `cell_coeff_jet_ab_match_fd_932` (1e-4) and
  `runtime_jet2_algebra_matches_finite_differences_932` (1e-4 absolute). The
  substrate (`Jet2`, `RuntimeJet`, `cell_base_moment_jets{,_moving}`,
  `cell_coeff_jet_ab`) is the test-side oracle that
  `moving_edge_leibniz_tracks_boundary_flux_932` imports. Its implicit-lift test,
  `runtime_jet2_implicit_lift_matches_analytic_ift_932`, compares against the
  analytic IFT at 1e-12.
- **The production BMS FLEX lowering** is gated by the high-order Richardson
  verifier `production_flex_grad_hess_matches_independent_fd_{score_warp,link_dev}_932`
  (gradient 1e-7, Hessian 1e-5 relative), plus the exact rigid and empirical-rigid
  oracles recorded above.
- The substrate's `e^{−Δq}` expansion mirrors the survival flex
  `survival/marginal_slope/timepoint_exact/flex_jet.rs::base_moment_jets`. Its
  percent-level finite-difference bars (`base_moment_jets_{first,second}_derivative_matches_fd_932`:
  1e-5 and 2e-4 relative, first and second order only, one cell) are replaced by
  `base_moment_jets_match_exact_theta_derivatives_through_order_five_932`. The new
  test checks every order the builder supports, one through five, at
  `1e-10·(1 + |oracle|)` on sextic, quartic, narrow and semi-infinite cells. Its
  oracle substitutes `z = z_L(θ) + t·(z_R(θ) − z_L(θ))`, carries a truncated Taylor
  series in θ through `(z_R − z_L)·zⁿ·e^{−q}` and integrates on a test-local
  Gauss–Legendre rule, so it shares neither the jet algebra, the `e^{−Δq}` closure
  nor the edge sliver with production. A one-part-per-million velocity corruption
  must fail every order.

### Live-family derivative census: where each joint Hessian derivative comes from

This covers the first requirement row, one row likelihood definition for every
live family. The census follows each production
`exact_newton_joint_hessian_directional_derivative` (and each `RowKernel`
third/fourth surface) to the expression it differentiates, reading main at
`89a782786`.

- **Derived from one row expression** (a `row_program!` lowering or a
  `JetScalar`/`RuntimeJetScalar`-generic body):
  - `EventHistoryFamily`: `evaluate_generic::<OneSeed<0>>` / `TwoSeed<0>`.
    spec-derivs (#2901 V1, SPEC rule 1) is measuring this path before choosing a
    Speed Gate or closed forms.
  - `BernoulliMarginalSlopeFamily`: rigid rows through `BernoulliRigidRowKernel`,
    flex rows through the runtime-jet flex program (`flex_row_program.rs`).
  - `BinomialLocationScaleFamily`: the order-2, third and fourth surfaces of
    `binomial_ls_row_program` (`ad41db449`), read by
    `binomial_location_scale_row_score`, `_row_hessian` and the first and second
    directional coefficients.
  - `BinomialLocationScaleWiggleFamily`: `BinomialLocationScaleWiggleRowProgram`.
  - `GaussianLocationScaleFamily`: `GaussianJointRowProgram`, lowered from
    `gaussian_normalized_row`. Its ALO replay reads the same row atom (`93ba47a33`).
  - `MultinomialFamily`: `directional_fisher_jet`.
  - `LatentSurvivalFamily`, `LatentBinaryFamily`:
    `latent_survival_row_primary_jet` with one- and two-seed backends.
  - `SurvivalLocationScaleRowKernel`: `sls_row_third_generated` /
    `sls_row_fourth_generated`; wiggle rows through `sls_row_nll_wiggle`.
- **1-D primitives** (derivative stacks of one scalar function, the kind this
  issue keeps): the GLM observed-information tower (`weight_ratio_tower`, above),
  the survival residual distributions (`residual_dist.rs`), the inverse links
  (`mixture_link.rs`), and `BoundedLinearFamily`'s per-coefficient bounded
  transform.
- **Hand-derived, checked against the single expression**:
  - The dispersion families (NB, Gamma and Beta in `gamlss/dispersion_family.rs`).
    `bf64d52d8` and `de14c2367` replaced the production jets with hand closed
    forms under SPEC rule 1. The generic row NLLs stay as test oracles
    (`dispersion_*_nll_generic`).
  - `TransformationNormalFamily`. Six producers in
    `transformation_normal/scop_curvature.rs` are written by hand for
    `f(β) = Σ_i w_i (½ h_i² − log h'_i)`: gradient, information, diagonal, matvec,
    and the first and second directional derivatives. Until `497956d82` they were
    checked only against one another.
    `ctn_scop_curvature_producers_match_exact_derivatives_of_the_likelihood_932`
    now checks all six against nested num-dual derivatives of that one expression,
    at 1e-11 relative. It first ran in pool job 578401 at `94de1340d` (EPYC 9534,
    HEAD guard clean) and passed. Before that it could not build: at `5e8436c44`
    gam-model-kernels failed (fixed by `d519f0c24`), at `3693c2c2c` gam-terms
    (fixed by `a8fbb3fa2`), and at `c74fbc04b` gam-solve (job 559612).

Not traced here: `GaussianLocationScaleWiggleFamily` and `SurvivalMarginalSlopeFamily`
past their `_for_specs` / flex dispatchers.

### Rigid BMS third cells on EPYC 9V74: the two programs are the same size

Nightly Speed Gates run 34689392350 at `f5693bb51` (gam-row-macros job, AMD EPYC
9V74) failed `RIGID-BMS-HAND-932`:

- `channel=third`: median ratio **0.988054**, wins 0.07, resolution 0.0048;
- `channel=third_full`: **0.958507**, wins 0.00, resolution 0.0063.

The newest completed run, 34706875875 at `a3f5987a9` (EPYC 7763), passed every
package.

**Emitted source.** Pool job 535606 at `5e8436c44` expanded the macro. Both
`generated_rigid_bms_third_contracted` (178 lines, 48 `*` operators) and
`generated_rigid_bms_third_full` (171 lines, 47) form the latent tower's ten live
slots, keeping the separable support of `q(η)` and the observed scale, then apply
the specialised root composition. That is the hand opponents' schedule.

**Instruction counts.** Pool job 539972 at `3693c2c2c` counted instructions in the
release timing closures of `generated_rigid_bms_matches_strongest_hand_932`, mapping
the ten closures in source order by their v0 disambiguators:

| channel | generated | strongest hand |
|---|---|---|
| order2 | 179 | 187 |
| third | 270 | 277 |
| fourth | 421 | 307 |
| third_full | 243 | 250 |
| fourth_full | 403 | 257 |

**Reading.** The two red cells race programs within 3% of each other in size. The
workflow sets no `target-cpu`, so every host runs this machine code, and whether
the generated arm wins depends on the host.

The generated side multiplies all ten slots by `outcome_sign`, where composing on the
unscaled binding needs only the stack powers `s^k`. This change has already been
measured and rejected. The dense compose arm of `row_program.rs` records that the
same rule emitted the same arithmetic minus the sign multiplies, yet lost 10–35% on
three hosts: LLVM scheduled the observed-scale inverse-power chain and its spills
ahead of the probit call. The order-2 emitter keeps the absorption because there it
wins on every host. Code shape therefore leaves no robust margin for the third cells
at parity.

**Decision (09-12, lead ruling).** `third` and `third_full` are now `not_slower`;
`order2`, `fourth` and `fourth_full` stay `faster`. A `not_slower` cell fails when
`median_ratio + ratio_resolution < 1`, where the resolution is half the central 90%
span of the paired ratios. This changes the contract only. It does not turn the
9V74 run above green: `third` gives 0.988054 + 0.0048 = 0.9929 and `third_full`
0.958507 + 0.0063 = 0.9648, both below 1, with wins 0.07 and 0.00. On that host the
generated arm is measurably slower than the strongest hand, so those two cells stay
red there until the generated program changes.

### Live-family derivative census, continued: the coefficient-space pullbacks

This closes the families the census above left untraced. It reads main at
`df35a6c97`. In each of these families the primary-space tower is derived
mechanically; what differs is how that tower is pulled back into coefficient
space.

- **`GaussianLocationScaleWiggleFamily`.**
  - Its predictor-space tower is generated: `gaussian_row_first_tower` and
    `gaussian_row_second_tower` read `GaussianJointRowProgram`, lowered from
    `gaussian_normalized_row`.
  - The pullback through the warp `q = q₀ + Σ_j βw_j B_j(q₀)` is written by hand:
    `gls_wiggle_first_directional_coeffs`, `gls_wiggle_second_directional_coeffs`
    and the dense `_from_designs` blocks.
  - No test compared it with the likelihood. The nearby FD gates cover the
    non-wiggle binomial family, and the wiggle ψ tests difference the dense Hessian
    against the ψ builders.
  - `5786dcd74` adds
    `gaussian_wiggle_joint_hessian_and_directional_derivatives_match_exact_derivatives_932`.
    It checks the dense H, `D_β H[u]` and `D²_β H[u, v]` against nested num-dual
    derivatives of `Σ_i w_i (½ (y_i − q_i)² / σ_i² + log σ_i)`, with
    `σ = LOGB_SIGMA_FLOOR + e^{η_ls}`, at 1e-10 relative. The warp is composed as its
    Taylor polynomial about the base index.
  - Reading the first-directional blocks against the chain rule (`h_mm`, `h_ml`,
    `h_mw`, `h_lw`, `h_ww = a_ww + a_wwᵀ + Bᵀ diag(H′_qq) B`) finds them consistent.
    The test first ran in pool job 578401 at `94de1340d` and passed.
- **`SurvivalMarginalSlopeFamily`.**
  - Rigid rows go through `SurvivalMarginalSlopeRowKernel`.
  - Flex rows read their primary tower from the flex jet evaluators
    (`flex_row_nll<J: FlexJet>`, `row_flex_primary_third_contracted_exact`).
  - The pullback over the dynamic q geometry is written by hand:
    `accumulate_dynamic_q_core_hessian`, the identity-block crosses, and
    `accumulate_timewiggle_directional_row`.
  - The time-wiggle arm is gated by
    `timewiggle_beta_hessian_second_directional_derivative_matches_finite_difference_2893`:
    resolving central differences at 1e-5 relative plus four times the
    uncertainty.
  - The flex no-wiggle arm had only build-once versus per-axis and
    subsample-operator identity checks. `63ad6877f` adds the same resolving gate,
    `flex_no_wiggle_beta_hessian_directional_derivatives_match_finite_difference_932`.
    It first ran in pool job 578401 at `94de1340d` and passed.
- **`BernoulliMarginalSlopeFamily` flex.**
  - The coefficient-to-primary map is linear: the marginal and slope designs, plus
    the score-warp and link-deviation coefficients as primaries (see
    `perturb_standard_normal_flex_states`). `D_β H` is therefore `Jᵀ T3[J u] J`
    with a constant `J`.
  - The primary channels are pinned by
    `standard_normal_flex_canonical_derivative_ladder_matches_vgh_t3_t4_932` and the
    Richardson verifier.
  - The coefficient-space directional surfaces are covered only by operator,
    batched and cache identity tests (`families_bms_joint_hessian_hvp_correction_tests.rs`).

### Affine cell moments: the anchor recurrence loses precision on finite cells

Every affine cell (`c2 = c3 = 0`) took its moments from `affine_anchor_moment_vector`. Its
truncated-Gaussian moments come from the upward recurrence
`T_n = a^{n−1}e^{−a²/2} − b^{n−1}e^{−b²/2} + (n−1)·T_{n−2}`, which amplifies the roundoff in
`T_0` and `T_1` like `(n−1)!!`. On a semi-infinite interval `T_n` grows at the same rate
and no precision is lost. On a finite interval inside the Gaussian bulk `T_n` shrinks
instead.

`affine_anchor_moments_match_quadrature_through_degree_34_932` (`654b1513c`) compares the
moments with a 20 000-panel Simpson reference. Pool job 580277 at `187735f30` (EPYC 9534)
measured relative errors on `[−0.3, 0.2]` with `η = 0.4 − 0.7z`: 1.4e-8 at degree 9,
2.0e-2 at 15, 8.3e4 at 21 and 2.1e20 at 34. The first cell's assertion stopped the run
before the off-centre and wide cells.

Production reads these degrees. The BMS row Hessians evaluate cells at degrees 15 and 21,
survival flex partitions at up to 32, and the order-five base-moment jets through `M_34`.
Interior partition cells whose score and link spans are locally linear are exactly affine.

`c7823aadd` evaluates finite affine cells on the certified Gauss–Legendre ladder and keeps
the anchor for semi-infinite and whole-line cells. Rigid rows have no split points, so
their single whole-line cell is unchanged. The GPU host classifier routes finite affine
cells to the device's non-affine branch, so the kernel source's `BRANCH_AFFINE` recurrence
is no longer dispatched. After the fix, pool job 584112 at `c7823aadd` (EPYC 7763) passed all
96 gam-model-kernels lib tests. The witness's worst relative errors are 3.1e-13 on
`narrow_bulk` (at degree 34), 4.2e-15 on the off-centre cell and 5.4e-13 on the wide cell.

Tails. The degree-34 witness measures finite cells only, and the tail tests before it stopped
at degree 4. `affine_anchor_tail_moments_match_quadrature_through_degree_34_932`
(`446423cae`) measures the cells the anchor still serves against a Simpson reference clipped
to `[−40, 40]` with 80 000 panels. Pool job 603819 at `7427b7e91` (acn67, EPYC 7763,
`v932tail.603819.log`) passed it. The worst relative errors are 2.3e-14 on a right tail from
−0.3, 1.1e-14 on a left tail to 0.2 and 1.2e-14 on the whole line. On the far right tail from
8 the error is 1.3e-12 at degree 0 and falls to 1.2e-13 at degree 34, which is the quadrature's
trend, not the recurrence's. The same job reproduced the finite-cell numbers above.
`d95dd4df6` deleted the kernel source's `BRANCH_AFFINE` define and validator arm, which no
host tag has emitted since `c7823aadd`.

## 2026-09-18 pass (i932)

Read on origin/main between `e7955a5704` and `cef0cf0c62`, and measured on MSI (EPYC 7763).

### Correctness witnesses at the tip

Every test named `*932*`, plus the non-932 witnesses the rows above name, in the test
profile with no patch:
- gam-models lib (job 1252877, `d094162c7d`): 151 passed, 0 failed;
- gam-math 14, gam-model-kernels 4, gam-solve 2, gam-gpu 1, gam-row-macros 13 and gam-sae 5
  (job 1255059, `d5827b0098`): 39 passed, 0 failed.

The run covers the rows marked "not run in the 09-12 pass":
- survival location-scale joint Hessian and cutover (37 `survival::location_scale` witnesses);
- implicit roots and moving boundaries (the empirical-rigid polynomial oracle, the
  fifth-vs-fourth tensor and `moving_edge_leibniz_tracks_boundary_flux_932`, whose
  second-order bar is still `1e-3`);
- the retired hand fourth-order oracle (the BMS flex V→t4 ladder and the survival flex
  contraction witnesses);
- SLS wiggle stages 2 and 3. `hand_sls_wiggle_row_third_matches_production_jet_932` and
  `hand_sls_wiggle_row_fourth_matches_production_jet_932` landed "not compiled or run
  before landing" (`e28cb2b3af`, `fc650e7fd8`) and now pass.

### Stable primitive algebra

The order-zero-to-four `ln Φ` table covered `x ∈ [−10, 2]` and divided each error by
`max(|reference|, 1e-3)`, so a right-tail entry of `1e-12` or less passed at any relative
error. `8d50993be7` holds every entry at `x = −100, −20, −8, 0, 8, 20` to the same `1e-11`
relative bound with no floor, against mpmath at 100 digits, and the subnormal second through
fourth derivatives at `x = 38.6` to four subnormal ulps. It was verified after landing by job
1256017 at `aaa0ddcfac`.

### Release receipts

The Speed Gates workflow has been `disabled_manually` since 09-12 20:55 CDT, so no workflow
runs the derived population, and the continuous-enforcement row cannot be met while it stays
off. `scripts/speed_gates.py --run` on MSI ran all 35 derived gates: gam-math, gam-row-macros
and gam-sae at `aaa0ddcfac` (job 1256028), gam-models at `cd4e4d3662` (jobs 1256607 and
1258687). 30 pass and 5 fail:

| gate | failing cells (`median_ratio`, wins) | contract |
|---|---|---|
| BINOMIAL-LS-HAND-932 | order2 0.663, third 0.547, fourth 0.781 (0.00) | faster |
| SLS-WIGGLE-HAND-932 | orders 2/3/4 at width 3: 0.084/0.125/0.120; at width 12: 0.029/0.041/0.051 (0.00) | not_slower |
| BMS-FLEX-CONTRACTED-932 | link-dev order 4 0.486 (0.00); score-warp order 4 0.998 (0.13) | faster |
| RIGID-BERNOULLI-VGH-932 | y = 1 0.777, y = 0 0.779 (0.00) | faster |
| BINOMIAL-Q-PRUNE-932 | 0.9917, resolution 0.0019 (0.00) | not_slower |

RIGID-BMS-HAND-932 passes every channel on this host (order2 1.019 through fourth_full 1.325),
as it did on EPYC 7763 before.

Receipt at `a7a0301949`. All 19 gam-row-macros cells pass (job 1279869), and 21 of the 23
gam-models gates pass (jobs 1283139 and 1283140). gam-math and gam-sae had no change since
the receipt above. The five cells that failed there now read:

| gate | `a7a0301949` | status |
|---|---|---|
| BINOMIAL-LS-HAND-932 | order2 1.022, third 1.031, fourth 1.027 | pass (`c7b453c3ec`, `a7a0301949`) |
| RIGID-BERNOULLI-VGH-932 | y = 1 1.012, y = 0 1.016 (wins 0.93, 1.00) | pass |
| BINOMIAL-Q-PRUNE-932 | 0.9991, resolution 0.0028 | pass |
| BMS-FLEX-CONTRACTED-932 | link-dev order 4 0.499 | fail |
| SLS-WIGGLE-HAND-932 | 0.028 to 0.129 | fail |

RIGID-BERNOULLI-VGH's red belonged to one build, not to the kernel code. The receipt binary
(`gam_models-a647504fed25e61e`, `cd4e4d3662`) and a build of the unchanged kernel at
`c7b453c3ec` ran interleaved on one host, three runs each (job 1282536). The old binary fails
at 0.762 to 0.789 and the new one passes at 1.010 to 1.043. The two measured functions
disassemble to equivalent code in the old binary: 155 and 151 instructions, one `sqrtsd`
each, one division each, the same call to `normal_logcdf_derivatives`. Raced against itself
in the same harness (job 1279199), each arm reads 0.996 to 1.002.

BINOMIAL-Q-PRUNE races two copies of one program. Production is the `Tower4` composition with
`d4 = 0`, and its opponent is the same composition reading `d4`; after inlining the fourth
channel is dead in both. A `not_slower` cell between them sits at the instrument's floor: an
identical function raced against itself in the rigid harness read 0.996 with resolution
0.0028, which alone fails `not_slower`.

BMS-FLEX-CONTRACTED link-dev order 4. The link-deviation row is a weighted compose sum over
the deviation coefficients (`bms/flex_row_program.rs`). The dynamic two-seed batch it is raced
against writes that sum as fused order-two blocks (`79ea1f0e7a`). The fixed `TwoSeed<K>`
specialization ran `JetScalar`'s default loop, one two-seed composition, product and sum per
coefficient. `TwoSeed<K>` now overrides it with eight order-two weighted sums at the one
composition point, joined by seven products. In three release runs (job 1286112) link-dev
order 4 reads 1.578, 1.582 and 1.658 (wins 1.00), score-warp order 4 1.002 to 1.018, and the
order-3 cells 1.51 to 1.70.

Binomial location-scale mechanism. Every production caller evaluated
`binomial_ls_row_program` at `δ = 0`. A `name: origin` primary role on `row_program!`, which
seeds the value with zero so the IEEE `0·x` terms fold, lifts order2 only to 0.766 (third
0.560, fourth 0.783; job 1261446). A `row_atom!` at-zero form of the same row, with the loss
as its Taylor polynomial in `q − q0` and an `active: bool` gate, gives 0.749 / 0.255 / 0.494
(job 1262339): `polynomial()` refuses a `Select`, so no channel under the gate was normalized.

The row is now `binomial_ls_row`, that at-zero `row_atom!` without the gate, called only by
wrappers that answer an all-zero loss stack with zero. Two generator changes came with it.
The at-zero normalization keeps exact rational coefficients, so a term that cancels exactly
is gone instead of leaving an `f64` residue (the third read `m4 * 1.3877787807814457e-16`
before, job 1263578). And each at-zero surface prefixes a constant it never reads with `_`,
so a surface takes the full declaration without an unused binding. Release gate at
`7c01c5dc95` (job 1269170, EPYC 7763): order2 1.005, third 0.963, fourth 1.025. Order2 and
fourth pass, and the third still fails `faster`.

Reading the emitted third (job 1270706): 16 multiplies, 6 additions and 4 negations, against
the hand's 17, 7 and 1. Job 1273155 raced rearrangements of a verbatim copy of that schedule
against the hand, two rounds each (EPYC 7763):

| third schedule | `median_ratio` |
|---|---|
| the copy as emitted | 0.957, 0.965 |
| its four channel signs moved onto the direction (two negations) | 1.033, 1.041 |
| the same, with the Horner chains reassociated to shorten the longest | 0.958, 1.001 |
| contracted first through the hand's `q_z = −r·z_t − q·z_ls` | 1.082, 1.065 |

The chain length is not what loses; the negations are. `row_atom!` now moves a contracted
surface's channel signs onto its first direction, one negation per axis shared by every entry,
when that strictly lowers the surface's count of distinct negations. The binomial third then
runs at 1.033 and 1.029 (wins 1.00), and every cell of the gam-row-macros suite passes in both
release runs (job 1274913). The cause-specific surfaces, whose negative channels would not get
cheaper, are emitted unchanged (job 1277751).

### Other rows, read

- Constrained Firth/Jeffreys: `c29a6ca084` arms both binomial location-scale families
  through the #979 lifecycle. `with_jeffreys_armed` keeps only `evidence.is_some()`
  (`location_scale.rs:2049`, `wiggle_custom_family.rs:8`), and neither family overrides
  `jeffreys_span_basis`, so an armed refit acts on the full span, which holds each family's
  likelihood gauge. No fixture arms and converges yet.
- GPU: the `survival_rowjet.rs` module doc says direct device tests cover ordinary and
  probability-tail rows. `c0a21b5540` deleted them, and only the two source-export tests
  remain. `calibration.rs:114-117` sets the row-kernel admission `row_kernel_min_n` from the
  XᵀWX crossover rows, and `bms/gpu/flex.rs:22` and `sae_rowjet.rs:1398` read it.
- Large-scale flex benchmark: `margslope_flex_large_scale_hv` times `cycle_capped_options(1)`,
  that is `outer_max_iter: 1` and one inner cycle
  (`tests/test_support/misc/margslope_flex_equivalence.rs:189-196`), not a converged fit.
- Moment degree: `FLEX_ORDER_FOUR_MOMENT_DEGREE` is still the literal 32. i2948's derived
  `4 + 6·Jet4::ORDER` passed its check and tests in job 1220614 and failed only the
  source-removal guard, for the moved const. It is unlanded. ad-outer's per-slot ladder
  certification (K) is designed, not landed.
