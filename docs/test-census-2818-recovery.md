# Historical regression recovery

The inventory in `test-census-2818-inventory.json` is a fixed source snapshot at
`7f141c5f430d0bb0aac2fe9e4e35385a470b681a`. Its `missing_untriaged` entries
describe that revision. This document records subsequent semantic recovery;
changing a historical status solely because a name reappears would confuse
source identity with executable coverage.

## Softmax entropy curvature: #1419 and #2339

The removed dense convenience methods are not needed by production. The exact
entropy HVP and majorizer value remain public in `gam-terms`; the active-entry
Hessian and analytic majorizer adjoint remain assembly inputs in `gam-sae`.
The recovered tests exercise those implementations directly. No deleted wrapper
or test-only production API is restored.

The public target `crates/gam-terms/tests/entropy_majorizer.rs` was published in
`e8fe18627b27b555be3e2be7be3066f2499b6258`. Its five tests executed on MSI:
**5 passed, 0 failed, 0 ignored, 0 filtered**, in 0.01 seconds after a 6.85-second
warm incremental build. Exact-commit CI also passed for
[source integrity](https://github.com/SauersML/gam/actions/runs/33945597611),
[runtime](https://github.com/SauersML/gam/actions/runs/33945597585), and
[compilation](https://github.com/SauersML/gam/actions/runs/33945597777).

| Historical test identity | Surviving contract and recovery evidence |
| --- | --- |
| `gershgorin_majorizes_entropy_where_fisher_does_not_1419` | The same-named public test checks the issue's analytic two-atom entropy counterexample through the actual HVP, demonstrates that Fisher fails, and verifies the Gershgorin quadratic bound. Executed in the five-test public target. |
| `smooth_gershgorin_majorizes_entropy_within_the_derived_budget_2339` | The same-named public test reconstructs Hessian columns through HVPs and checks majorization and the derived smoothing budget on 32 rows. The cancellation-aware roundoff allowance consumes at most 0.000093203 of that budget; the largest majorization deficit is 0.375903 roundoff units. Executed in the public target. |
| `smooth_gershgorin_is_degree_one_homogeneous_in_scale_2339` | The same-named public test executes the value contract. SAE's `smooth_gershgorin_adjoint_is_degree_one_homogeneous_in_scale_2339` executes the historical analytic-adjoint arm for every atom/logit pair and three power-of-two scale factors. |
| `smooth_gershgorin_weighting_routes_agree_by_scale_equivariance_2339` | The same-named public test compares the actual weighted trait route to assembly's folded-scale value route. The SAE adjoint homogeneity test also verifies that folding a positive power-of-two weight into scale gives the same derivative as post-multiplication, recovering the historical adjoint arm. |
| `smooth_gershgorin_is_exactly_zero_on_an_underflowed_atom_2339` | The same-named public test checks a zero-mass atom beside two active atoms, keeping nonzero curvature in the fixture. SAE's `smooth_gershgorin_adjoint_is_exactly_zero_on_an_underflowed_atom_2339` executes the historical analytic-adjoint arm while requiring finite, nonzero derivatives on the active atoms. |
| `gershgorin_majorizer_logit_derivative_matches_fd_1419` | The same-named SAE test checks all 16 active-entry derivatives against finite differences of the production radius; maximum error 1.398283e-10, maximum reference magnitude 0.8148608, against the original 1e-6 error bar. |
| `smooth_gershgorin_adjoint_is_continuous_across_a_zero_crossing_2339` | The same-named SAE test measures a hard-radius jump of 0.1708504. Production's derivative jump contracts from 0.001708415 to 0.0001708532 when the probe distance shrinks tenfold. |
| `smooth_gershgorin_adjoint_matches_fd_inside_the_smoothing_band_2339` | The same-named SAE test checks all nine atom/logit derivatives inside the measured 1.518764e-9 smoothing band. Maximum finite-difference error 4.228824e-6 against the original 1e-3 error bar. |

The five SAE tests are in
`crates/gam-sae/src/manifold/softmax_entropy_majorizer_tests.rs`, registered
through the production assembly module. Their finite differences are test
oracles only. The seam fixture isolates an off-diagonal zero crossing in a
three-atom Hessian whose remaining row stays nonzero, and measures the smoothing
band from that row's own curvature. A hard-absolute-value counterfactual must
exhibit a jump while the production derivative's jump contracts with probe
distance. A two-atom crossing cannot provide this oracle: gauge invariance makes
the entire row vanish there.

All five executed successfully in the combined MSI SAE selection (49 tests,
4.21 seconds runtime). That combined command exited nonzero: 47 passed and two
separate exact-A tests failed. The five-test recovery result does not assert
that the entire selection passed. The executed source Git blob identities are
`a6fa7f6ef107b9de2e889017241361814e0cfd60` for the leaf containing the module hook
and `64f96e2407713cab8914f2b3a353490c1f88adbc` for the sibling test file. Evidence
is recorded in MSI's `.buildd/exact-a-block-majorizer-combined.log`.

## Active assembly entries: #1410

Two historical #1410 identities compared active entries bit-for-bit to the
deleted dense convenience methods, which shared the same expressions. Both
names are now restored with independent oracles in the same SAE sibling target:

| Historical test identity | Executed replacement contract |
| --- | --- |
| `active_softmax_dense_entropy_hessian_entry_matches_dense_block_1410` | Reconstruct the dense Hessian from the public entropy HVP, then compare every active entry. Three 48-atom fixtures cover varied, uniform, and underflowed probabilities. Maximum error 1.387779e-17, maximum reference 0.08343182, against the 2e-13 roundoff allowance. |
| `active_softmax_majorizer_logit_derivative_matches_dense_1410` | Compare the active analytic adjoint to finite differences of the public dense majorizer value, covering every atom and three logit probes on each of three 40-atom fixtures. Maximum error 1.514916e-10, maximum reference 0.3798330, against the 1e-6 finite-difference allowance. |

Both comparisons require nonzero reference values. The HVP calculation does
not share the leaf's entry expression; the derivative oracle does not restore
the deleted derivative wrapper. Bit-for-bit identity between duplicate
expressions has been replaced by the underlying live operator contract, with
roundoff/finite-difference allowances appropriate to independent arithmetic.

All seven SAE sibling tests executed successfully with source blob
`921394d1e849da5c4b778c6c573217600869b7cb`, including the five previous derivative
tests. The combined command ran 21 tests in 3.54 seconds: 19 passed and two
failed, including an unverified #2080 fixture premise. This is evidence for the
seven named passing tests, not an all-green combined suite. The log is
`.buildd/exact-a-obb-majorizer-adjoint.log` on MSI.

## Resource governor: #2317, #2684, and #2702

Both historical resource-governor pins are restored in
`crates/gam-runtime/src/resource.rs`, using the observation constructor and
reservation ledger that production uses. No removed observation fixture or
budget accessor is reintroduced.

| Historical test identity | Executed replacement contract |
| --- | --- |
| `literal_unlimited_cgroup_defers_to_host_available_memory_2317` | An unbounded controller preserves host availability and capacity, including typed provenance. Reserving the exact 3 GB host budget succeeds, one further byte is refused, and dropping the reservation restores the budget. An unbounded controller does not make host capacity unbounded. |
| `a_cgroup_at_its_limit_moves_neither_the_budget_nor_the_materialization_cap_2684_2702` | A 6 GiB controller is observed idle, 53,248 bytes below its limit, and fully charged. Its 4.5 GiB budget and materialization ceiling stay fixed while observed availability changes. The actual 1,024-byte coefficient-SE reservation succeeds and releases at each load; an 8 GiB request is refused. A 1,024-byte controller retains a 768-byte cap and refuses the 28,800-byte design. |

The tests reserve ledger entries, not actual gigabytes of memory. Both executed
successfully on MSI: **2 passed, 0 failed, 0 ignored, 99 filtered**, in 0.00
seconds after an 18.39-second build. The executed source blob is
`987091257b1c47cb3cde144297882e16b1c87a36`; the log is
`.buildd/runtime-recovered-pins-2818.log`. A fresh governor's actual remaining
budget replaces the deleted convenience accessor, and admission/refusal
assertions independently exercise that budget instead of merely restating the
derivation.

## Full-basis probe adjoint: #2080

`sae_logdet_theta_adjoint_from_probes_matches_dense_softmax_2080` is restored in
`crates/gam-sae/src/manifold/tests_deflated_from_probes_2712.rs`. It compares the
production dense adjoint, using the materialized joint selected inverse, to the
production probe adjoint with a complete deterministic basis of reduced-Schur
probes. Both contract the same majorizer operator. No iterative inner solve,
deleted dense wrapper, or stochastic convergence assumption is needed for this
algebraic identity at a fixed state.

The fixture explicitly places periodic coordinates in the positive-curvature
quarter of their ARD prior and refreshes the production basis. It then asserts
that the actual factor has no deflated row directions. The historical #2080
gate admitted this undeflated regime; the separate historical #2712 deflated
adjoint gate is verified below on a derivative-sensitive fixture. The existing cold state's two spectrally deflated
rows did not resolve that latter contract: its deflation-aware and blind
adjoints differed by only 3.552714e-15. Merely counting those rows would have
claimed coverage of a derivative the fixture could not distinguish.

The restored #2080 pin passed with adjoint magnitude **14.29163** and maximum
dense/probe error **2.664535e-15**, retaining the tighter 1e-10 comparison bar.
Removing the Schur-inverse contribution produces a **0.6203322** discrepancy;
the test requires this counterfactual to separate by more than its error bar
and by at least 1,000 times the actual parity error. This ensures the inverse
probe contractions make a measurable contribution.

The final combined selection passed **2 tests, 0 failed, 0 ignored**, in 2.16
seconds, including this pin and the separate ordered-Beta--Bernoulli adjoint
gate. The executed source blob is
`ed719024f987eb3dbdf54afb7078c715dd83906d`; the MSI log is
`.buildd/exact-a-obb-scalar-and-2080.log`. The existing #2712 selected-inverse
reconstruction test also passed in the preceding selection, resolving
off-diagonal reconstruction to 1.11e-16 on two spectrally deflated rows. That
block-reconstruction result is distinct from the deflation-adjoint sensitivity
requirement verified below.

## Public basis geometry and storage: #2315 and #2684

`crates/gam-terms/tests/basis_scale_recovery.rs` restores
`sphere_constant_curvature_and_pca_obey_their_non_euclidean_gauges_2315` through
the live public builders. It checks spherical Wahba and harmonic designs and
penalties in degrees versus radians; constant-curvature designs, penalties and
kernel jets under coordinate scales 1e-9, 1 and 1e9; and PCA designs and penalties
under inverse loading rescaling. Curvature rescales as inverse squared length,
so kernel value, first curvature derivative and second curvature derivative
carry length powers 1, 3 and 5. The live joint `constant_curvature_kernel_psi_jets`
carrier replaces the deleted curvature-only helper. Nonzero derivative witnesses
prevent a zero implementation from satisfying scale equivariance vacuously.

The #2315 test passed in **0.02 seconds** in
`.buildd/spatial-basis-2827-forward-tests.log`, with first/second derivative
witnesses **0.03467601 / 0.009046658**. Source blob:
`61e1e659788d10661315071a205035c7584bcde8`. The same Cargo invocation subsequently
failed the initial #2684 oracle; this is an individual passing test, not a claim
that the original combined run succeeded.

`crates/gam-terms/tests/basis_storage_recovery.rs` restores
`the_storage_route_changes_how_the_basis_is_carried_not_which_basis_it_is_2684`.
It builds the original 300-row, 12-center pure Duchon fixture under a simulated
6 GiB job policy, asserts actual dense versus operator storage, checks all 11
coefficient basis actions, and compares every active penalty's source, nullity,
shape and entries. The restrictive arm is exercised through operator actions
without forcing its design to materialize.

The initial test incorrectly compared raw coefficients from two independently
chosen cold charts and failed at column 3 (operator -0.06297778 versus dense
0.08153217). The data-metric radial eigensystem and final identifiability
nullspace are fit-time coefficient charts. The corrected storage test replays
both actual transforms from dense fit metadata into the operator build, using
the production prediction/replay contract. No comparison tolerance changed.
This establishes representation parity for the same basis chart; independent
cold-chart reproducibility and output-rotation fit equivariance remain separate
obligations and are not discharged by this test.

The corrected #2684 target passed **1 test in 0.16 seconds**, after a **4.62-second**
warm integration build. All 300-by-11 design actions agreed within
**4.649059e-16**, and active penalty entries agreed exactly. Log:
`.buildd/basis-storage-frozen-recovery.log`; source blob:
`8de3fcfab0373020ad262695fb3758eeadb42abe`. The umbrella scanner passed with
explicit filesystem membership for both new targets; its report was empty.
The runs use the forward-synchronized published Duchon basis implementation.
They do not execute unrelated stale local term-collection gauge tests.

## Curvature estimate support: #2687

`crates/gam-geometry/tests/curvature_support_recovery.rs` restores both historical
pins through public `profile_ci_walk`, without reinstating the removed
`is_railed` convenience accessor:

| Historical identity | Current asserted contract |
| --- | --- |
| `a_monotone_criterion_rails_kappa_hat_and_the_walk_declares_it_2687` | A decreasing profile reports the upper rail when the box moves to 1.389, 2.78 or 40; the mirrored increasing profile reports the lower rail. Point-estimate labels and the corresponding open CI endpoints agree. |
| `an_interior_optimum_is_not_declared_railed_2687` | The fixed quadratic optimum -0.37 is interior in a wide box, with closed symmetric CI endpoints. Moving only the box's lower end onto the same optimum changes its provenance to the lower rail, while preserving the unconstrained upper CI endpoint. |

These are inference-layer contracts for a supplied profile, not claims that an
outer fit has correctly profiled its nuisance parameters. Both tests passed in
**0.00 seconds**, after a **4.43-second** warm integration build. Log:
`.buildd/curvature-support-recovery.log`; executed test source blob:
`f6a703e9cc9277e70d063281c826d6185bd369a0`. The production curvature-estimand source
was `f899e14cf6ea6b6fe5f90910a72934c63ea7af76`, exactly the published `684f392d`
source. No production implementation changed. The umbrella scanner passed with
an empty report and explicit membership for the new integration target.

## Coordinate collapse, seeded controls and local transitions: #2691, #2250, #2280

The next batch restores eight historical identities and adds one support-aware
negative control. All **9 tests passed** in MSI log
`.buildd/exact-beta-third-owner-recovery.log`. The combined selection had
**21 passes and 1 separate new Threshold derivative failure**, 22 tests in
3.65 seconds; the full selection was not green. The umbrella scanner passed
before the build, with explicit membership for both new test-only siblings.
They carry file-level `#![cfg(test)]` as well as their parent module's gate so
the lexical scanner sees their test context without following module edges.

The four historical #2691 names are restored in
`coordinate_fidelity_recovery_tests.rs`, through surviving weighted circular and
interval classifiers with unit support masses:

| Historical identity | Asserted current contract |
| --- | --- |
| `a_constant_coordinate_is_collapsed_not_continuous_2691` | Exact constants and the measured approximately 1e-14 coordinate spread report `Collapsed`, zero effective rank, zero anchors and the collapsed label. |
| `a_narrow_but_resolvable_arc_is_still_continuous_2691` | A 70-point arc of width 0.12 reaches an actual occupancy model. As in the historical body, its BIC-winning rung is deliberately not prescribed by the test's name. |
| `uniform_and_discrete_occupancy_survive_the_collapse_guard_2691` | Uniform and seven-anchor weekday-shaped support survive the collapse guard and are not indeterminate. |
| `collapse_across_the_wrap_point_is_caught_on_the_circle_2691` | A near-full raw range occupies a collapsed circular arc, while the same points span a genuine interval support. |

The new `zero_mass_outliers_do_not_hide_coordinate_collapse_2691` additionally
asserts that zero-mass distant rows cannot hide collapse, and that assigning
those same rows positive mass changes the extent verdict. No deleted unweighted
classifier wrapper was reinstated.

`matched_spectrum_gaussian_preserves_pc_scales_and_is_seeded_2250` is restored in
`null_battery.rs`. The live full-covariance Gaussian generator is given the
historical 8,192-by-3 orthogonal Fourier PC fixture. Its population means and
standard deviations are known analytically, so the oracle does not reuse the
generator's moment accumulation or covariance factor. Repeating the seed must
reproduce every draw, and changing the seed must change the generated control.
Maximum mean error was **1.440434 standard errors** and maximum relative scale
error **0.002321333**, within the original **4-standard-error / 5%** bars. The
deleted diagonal-only generator remains removed.

Three #2280 names are restored in `local_chart_recovery_tests.rs`. Their oracle
composes the live `ChartTransition` rotations/signs on a genuine shared-row
triple; it does not restore deleted rotation/cocycle convenience methods.

| Historical identity | Executed result and unchanged bar |
| --- | --- |
| `swiss_roll_charts_injective_and_cocycle_closes_2280` | 36 charts; positive lower projection stretch and captured fraction above 0.7; rotation-cocycle defect **0.1398487 < 0.5**, sign product +1. |
| `embedded_plane_cocycle_closes_to_rounding_2280` | 24 exact-plane charts; captured fraction above 1-1e-9 and stretch within 1e-6 of unity; defect **2.874508e-16 < 1e-8**. Live affine transition composition also agrees within **2.220446e-16**, and the direct transition reproduces the observed target-chart coordinate. |
| `sphere_charts_injective_and_orientable_2280` | 34 spherical-band charts; positive lower stretch, observed orientability and sign product +1; defect **0.7312095 < 0.75**. |

The historical spherical fixture has its polar caps removed. Its local
transition and observed-orientability evidence is not a closed-sphere homology
or calibrated population-topology certificate. These tests also do not prove
atlas recognition, held-out unrolling quality or MDL promotion.

The other three #2280 identities were restored in `local_charts.rs` together
with the `LocalAtlas::co_collapse_candidates` query they exercise (#2829):
`co_collapse_flags_duplicate_charts_2280`,
`co_collapse_thresholds_bracket_the_gate_2280`, and
`co_collapse_spares_healthy_swiss_roll_atlas_2280`. `e6fd4251e` then deleted the
query, its `CoCollapseCandidate` report and all three pins, because nothing but
those pins queried it.

Executed source blobs:

| Source | Blob |
| --- | --- |
| `coordinate_fidelity.rs` | `610f3bf531e86a0cf1191dbc3073e7fafe504166` |
| `coordinate_fidelity_recovery_tests.rs` | `04edc62858a0520c707aaf882085f8d0cbda31bc` |
| `null_battery.rs` | `ced61aeefa187acba048a87e70ef40d86d95384c` |
| `local_charts.rs` | `5790dc048554626dec1313283095c1ae5aa59275` |
| `local_chart_recovery_tests.rs` | `74ff6fdc454eddfaf6d0df8f39ece919ca2c6d90` |

The three production-owner files matched published main before these test-only
additions and formatting; no production implementation changed in this batch.

## Deflation-sensitive probe adjoint: #2712

`sae_logdet_theta_adjoint_from_probes_matches_dense_on_deflated_rows_2712` is
restored in `crates/gam-sae/src/manifold/tests_deflated_from_probes_2712.rs`.
It checks the production majorizer adjoint at a fixed state, with a genuine
spectrally conditioned factor and an independently constructed complete basis
of Schur probes. The fixture's weak positive periodic ARD curvature lies inside
the smooth-clamp tail: its value is below the spectral floor while its coordinate
derivative remains resolved. The phase is derived from the production clamp
temperature and spectral floor, without a parameter search or cache mutation.

Zero decoded tangents intentionally isolate that prior contribution. They also
produce no decoded-derivative gauge, which initially left spectral discovery
disabled in the direct test. The corrected fixture uses
`ensure_row_gauge_deflation_for_quasi_laplace`, the same production installer
used before frozen-state evidence factorization. This is an algebraic comparison
of a conditioned majorizer; it does not certify a fitted state or an exact-A
maximum.

The final test passed on **5 spectrally deflated rows**, with adjoint magnitude
**3.033522**, maximum dense/probe error **1.776357e-15**, and discrepancy
**5.654301e-4** when the dense route omits the Daleckii--Krein correction. The
1e-10 comparison bar is unchanged. The counterfactual must separate by more
than that bar and by more than 1,000 times the actual parity error, so agreement
cannot pass on the cold fixture's previously unresolved correction.

All **3 tests in the module passed** in the combined MSI selection: the new
deflation-adjoint pin, the #2080 undeflated pin and the existing off-diagonal
selected-inverse reconstruction pin. The whole selection had **38 passes and
7 separate block-solver failures**, 45 tests in 0.62 seconds, so it was not an
all-green repository run. Executed source blob:
`241a79fde98e1afe620e8519c99f327e06d8f207`; log:
`.buildd/exact-beta-prepared-block-2712.log`. The umbrella scanner also passed
with this source present. The new adjoint witness resolves the weak ARD
conditioning contribution; it does not replace separate outer-gradient,
assignment-strength trace or off-diagonal spectral-rotation contracts.

The historical `zz_measure_deflation_correction_size_2712` was a diagnostic,
explicitly reporting rather than asserting correction size. It printed NaNs
when either adjoint evaluation failed. Its test-acceptance purpose is replaced
by the asserted non-vacuity control above, so the print-only experiment is not
reinstated as a passing regression gate. Its old three-fixture measurements
remain historical observations; this recovery makes no new execution claim
about those removed fixtures.

## Channel-aware canonicalization: #1590

The public integration target
`crates/gam-identifiability/tests/canonical_recovery.rs` restores two historical
contracts through `canonicalize_for_identifiability_with_operating_scalars`,
with no deleted canonicalization wrapper or production change.

| Historical test identity | Executed replacement contract |
| --- | --- |
| `canonical_dead_column_callback_block_is_not_reduced_1590` | The channel-aware audit detects an identically zero column in each cause's design, while each family-owned callback retains its raw coefficient width. Nonzero coefficients lift identically; returning zeros would fail this stronger identity check. |
| `penalty_covered_competing_risks_redundancy_canonicalises_cleanly_1590` | Two channel-aware blocks have eight coefficients but six independent likelihood directions. Canonicalization retains seven coefficients, preserves all six data directions, and pulls back each identity penalty to an identity on the retained coordinates. The remaining data-null coefficient stays identified by its penalty. Independent Gram eigendecompositions verify the raw and retained ranks. |

Both tests passed on MSI: **2 passed, 0 failed, 0 ignored, 0 filtered**, in 0.13
seconds after an 8.13-second warm integration-target build. The executed test
blob is `79ccbc0b7204c2a0bb58ffe985ca8c1e883f622b`; the log is
`.buildd/identifiability-recovered-pins-2818.log`. The production audit/kernel
blobs match published main `1df6210f50aaab1f49366ac18afcf3b98fee28c2` exactly.
The canonicalization source on MSI differs only by four already-published
`#[cfg(test)]` #2748 gates absent from that worktree; its production code is
identical. This integration run therefore supplies no execution evidence for
those four separate unit tests.

The publication's compile policy gate then caught an ignored callback state
parameter that the member-crate integration build does not check. The fixture
now validates the coefficient width whenever an operating point is supplied;
the coefficient-independent row operator's explicit empty-beta request is
also accepted. Both pins passed again: **2 passed, 0 failed, 0 ignored, 0
filtered**, in 0.12 seconds after an 8.08-second warm build. The corrected source
blob is `cd9b98dcd6ebd3232df6a373d0da3ca732c22e1b`; the log is
`.buildd/identifiability-recovered-pins-width-2818.log`.
The umbrella scanner, freshly compiled from `build.rs` blob
`1e65d2a548b4722ff329321ff6fc8271d61180db`, also passed on that MSI worktree.
Its source walk explicitly included the integration target. The separate
tracked-file infrastructure and line-count checks still use the MSI index;
the publication's CI gate supplies the exact committed-index verdict.

## Nested derivative algebra: #932

The public integration target `crates/gam-math/tests/nested_dual_recovery.rs`
recovers four contracts without adding a production constructor or test-only
trait bridge. A single smooth program is evaluated through the live `JetField`
implementations of nested second-order `Dual2` and the dense fourth-order
`Tower4`; their derivative propagation orders are independent.

| Historical test identity | Executed replacement contract |
| --- | --- |
| `nested_dual2_reproduces_tower4_channels_932` | All nine represented channels agree with the dense tower at four points: value, both first derivatives, three second derivatives, two mixed third derivatives and the mixed fourth derivative. |
| `nested_dual2_directional_matches_tower4_contraction_932` | Independently seeded arbitrary directions agree with explicit contractions of the tower's first, second and fourth derivative tensors at three points. |
| `nested_dual2_seed_swap_symmetry_932` | Swapping inner and outer seeds preserves all nine channels under their corresponding index permutation, extending the historical four-channel check. |
| `nested_dual2_channels_from_channels_roundtrip_932` | The deleted `from_channels` convenience constructor has no surviving API contract. `nested_dual2_channels_follow_polynomial_derivative_order_932` instead evaluates a polynomial whose analytic derivatives are exactly 1 through 9 in the documented channel order. This checks layout and repeated-derivative factorials independently of a getter/setter roundtrip. |

All four tests passed on MSI: **4 passed, 0 failed, 0 ignored, 0 filtered**, in
0.00 seconds after a 4.08-second warm integration build. The nine-channel maximum
relative error was **3.718561e-16**, with mixed fourth-derivative magnitude
**2.665960**. Directional contraction error was **2.220446e-16**, with fourth-order
contribution **0.7515526**. Both fourth-order witnesses must exceed 0.1, and the
comparison bar remains 1e-12. The polynomial channels equal `[1,2,3,4,5,6,7,8,9]`
exactly. Executed source blob: `3672204465bfbec56340c0e5eceaad39c35875b4`;
log: `.buildd/nested-dual-recovered-pins-2818.log`.

The exercised `nested_dual.rs`, `jet_tower.rs` and `Tower4`'s `JetField`
implementation match published main `90c4a32a04e3845d310fba9cb7b134e3ff4e21e8`.
The MSI `jet_scalar.rs` also contains unrelated, preexisting differences in
runtime weighted-composition methods; this target provides no verification of
those separate methods or their tests.
The freshly sourced umbrella scanner also passed on the MSI worktree with the
new integration target present; the same committed-index limitation described
above applies to its infrastructure and line-count checks.

## Typed optimizer failures: #2658

`crates/gam-problem/tests/error_source_recovery.rs` restores
`fatal_optimizer_evaluation_retains_exact_typed_source_2658` through the public
`OuterObjectiveErrorSource` enum and its live typed downcast. It preserves the
optimizer producer's fatal verdict, original context, and every recorded
inner-solve field: 17 cycles, residual 3.5, tolerance 0.25 and dimensions
4/3/1. A second outer orchestration wrapper must retain that original boundary
and its optimizer source. The deleted `objective_error` convenience accessor
is not reintroduced.

The verified MSI run executed **1 passed, 0 failed, 0 ignored, 0 filtered**, in
0.00 seconds after a 4.50-second warm target build. The executed source blob is
`498d6279aa768e1b5b655e36b44cd3d82c6d17ee`; the log is
`.buildd/typed-error-recovered-pin-2818.log`. An earlier stdin upload was consumed
by the wrapper's node-selection command and produced an empty test target;
its zero-test run was rejected as evidence. Standard file transfer, actual
compute-source hashes and the explicit one-test count establish this result.

The exercised error-boundary implementation matches published main
`b87e911ca431a9318cb0968bff0904e7809ea043`. The MSI custom-family error source has
an unrelated, preexisting missing terminal-reason variant; the tested
`InnerSolveNotConverged` fields are identical. This pin supplies no
coverage of that separate terminal-reason variant.
The umbrella scanner also passed with the actual nonempty integration target
present. Its filesystem source walk includes the target; the previously noted
committed-index limitation remains for the infrastructure and line-count checks.

The remaining historical inventory is still open. A successful source census,
or the restoration of these few contracts, does not establish that all 303
historically deleted pinned identities have been recovered or retired.

## Boundary backoff: the #2695 contract was deliberately retired

The upstream retirement decision for
`a_clipped_step_stops_one_tolerance_short_of_the_face_2695` remains in force.
That test required a clipped step to stop one `PRIMAL_FEASIBILITY_TOL` before
its blocking face, independently of the direction's length. Production later
rejected that behavior: the retreat left the row outside the tighter active-face
tolerance, preventing it from entering the working face.

The current `apply_feasible_step_boundary_backoff` in
`crates/gam-models/src/marginal_slope_shared.rs` returns the clamped step ratio
without a retreat. A clipped step lands on its blocking face; a direction that
pushes out through an already-active row receives a zero feasible fraction and
requires a projected direction, rather than another shortened step. The module
documents both the former fractional retreat and the one-tolerance retreat as
rejected policies.

The surviving checks
`feasible_step_fraction_refuses_a_non_finite_direction_2721` and
`feasible_step_fraction_admits_a_sub_tolerance_drift_off_an_active_row` exercise
the current behavior with positive controls. The old `unit_box` fixture remains
available, so the missing historical test is not explained by a missing helper.
Its status is **retired**, not recovered: reintroducing its original assertion
would require the obsolete acceptance rule to return. This preserves the
explicit semantic retirement previously recorded on main.

## Current acceptance recovery: September 7

A fresh MSI source walk of `crates`, `tests`, and `src` used the census's Rust
lexer on the current working tree. After adding the four structural-coordinate
pins and the Trace consumer pin below, **243 of the 303 historical identities
were absent across 106 historical source paths**. The earlier count of 248
predated these five declarations. This is a working-source observation, not an
immutable commit census or evidence that those declarations executed.

| Historical contract | Current production acceptance seam | Evidence and remaining work |
| --- | --- | --- |
| Four structural-coordinate #2748 pins in canonicalization | External `gam-identifiability` integration target `structural_coordinates_2748` calls `canonicalize_for_identifiability_with_operating_scalars`, with declared spanning/structural coordinates. | **4 passed.** Tests check priority-directed dimension reduction, structural coefficient preservation, independent reduction of another block, and refusal of a wrong-length coordinate list. |
| Four measured-span #2612 pins | External `gam-solve` target `measured_span_2612` calls `under_identified_subspace_in_metric` with explicit identity or congruent shear metrics. | **4 passed in 0.022 seconds**, after a 3.70-second warm build. Tests check the strict one-observation boundary, empty bounded span, disagreement with the penalty kernel in both directions, and physical-span preservation under both signs of a nonorthogonal shear. A wrong-metric control must change the selected dimension. |
| `softmax_trace_whitening_prefold_matches_dense_adjoint_2333` and the four later deflation-fold checks | `construction_row_jet_logdet_channels::tests_trace_whitening_2333` exercises the live Trace consumer and production deflation contraction. | **5 passed in 0.05 seconds.** The first consumer run correctly failed its branch-activation assertion because the cold fixture had no spectral deflation. The revised state has one exactly saturated softmax row, which supplies a genuine null logit direction; the other rows retain both live atoms. The consumer records one spectrally deflated row. |
| `fully_degenerate_cluster_diagonalizes_direct_e_diag_2267`, `nearly_degenerate_distinct_spectrum_preserves_eigenpairs_2515` | `exact_pencil::pencil_tests`, reached through production pencil pricing. | The original helper-only tests moved to determinant/gradient and eigenpair checks on the actual production pencil. The obsolete `cluster_stable_eigh` helper was removed. These are explicit contract moves, not disappearance based on private symbol reachability. |
| Five historical cofit #2023 pins | Active `tiered::fit_tiered` versus the former `cofit_arrow` and `sparse_dict::cofit` bridges. | **Retired with their subject (#2829).** `fit_tiered` carries the contracts on the live route: match-or-beat against the pure-linear tier and a recurred, certified fixed point are asserted by `tiered_curved_refinement_is_certified_and_records_promotions`, and the budget refusal by `insufficient_inner_budget_returns_error_instead_of_a_tiered_report_2023`. The producer-less cofit configuration/report types are deleted. |
| Three co-collapse #2280 pins | Former `LocalAtlas::co_collapse_candidates`. | **Retired with their subject (#2829).** Restored with the query, then deleted by `e6fd4251e` together with the query and its `CoCollapseCandidate` report, because nothing but these pins queried it. |
| Solved-mode response #2765 and terminal exact-curvature scheduling #979 | Current survival marginal-slope mode response and spatial outer driver. | **Unresolved.** A doc-comment-only historical test file is not coverage; local Hessian derivatives do not establish a fitted mode's response, and an outer-Hessian check does not establish terminal scheduling. |

The four Trace fold checks passed within a **12-test run: 10 passed, 2 failed**,
in **0.187 seconds**, logged at
`bench/measurements/issue_triage_20260907/sae-focused.log`. The other failure was
the separately tracked #2825 epoch-quality acceptance. This run used the current
MSI working source, including preexisting and newly authored changes; it does
not certify a published commit or the failing Trace consumer. The fold checks
span every symmetric derivative basis direction, require a nonzero correction,
and cover gauge-only, unit-deflated, floor-clamped, and degenerate split spectra.

The subsequent fresh Trace binary passed all five tests on one CPU. The dense
joint-versus-Trace maximum gap was **2.55440113505756e-12** and the coordinate
gap **3.979039320256561e-13**, against adjoint magnitude **199.93419830762554**.
Both use the unchanged `1e-12 * (1 + magnitude)` bar; repeated joint results
must be bit-identical. Log: `bench/measurements/issue_triage_20260907/trace-cached.log`.
The fresh SAE test build used 16 code-generation units and eight assigned CPUs,
finishing in 79 seconds; reusable nextest metadata allows subsequent tests to
run from that executable without recompilation.

The four measured-span checks are recorded in
`bench/measurements/issue_triage_20260907/measured-span-focused.log`, with executed
test-source SHA-256
`5255905baa6554012eec8cbed4343b6a8106c34a09d64e0fbe21cd299d4ee35b`.
The structural-coordinate source SHA-256 is
`474f2d6431035e3d24d4b8d0c083504f6677cbaa5cba4a958b3cf2588bf87ecf`;
its four passes occur in `bench/measurements/issue_triage_20260907/public-api-focused.log`.
That six-test run also had one public API fixture failure: its expected
unridged least-squares coefficients omitted the solver's declared fixed
stabilization ridge. The corrected public API target uses already standardized,
orthogonal columns and checks the exact equations
`(6 + 1e-8) beta = [12, 18]` at a stricter `1e-12` bar. Both of its tests then
passed in `bench/measurements/issue_triage_20260907/public-api-final.log`.

The largest still-unmapped numerical groups include twelve historical exact-A
#2515 checks, eight logdet-adjoint #2156/#2144/#2330/#2712 checks, seven curved
co-collapse #2027/#2132/#2082 checks, and nine outer-curvature invariance
#2676/#2748 checks. Their recorded historical names remain in
`test-census-2818-inventory.json`. A nearby test or closed issue is insufficient
to retire any of them: each requires a comparison of the exercised current
criterion, derivative, or acceptance decision and an executed replacement.

## Closed-issue pins rebuilt on live APIs: #1463, #2561, #2598, #2548, #1017, #2676

Landed in `5e1759ec9`. Ten historical identities deleted by `c0a21b554` are restored
under their original names. The production contract behind each pin survived the
sweep; what `d484a091a` removed was the convenience surface the pins called. Every
fixture is built inside its test, and no deleted helper is restored. Two of the
restored identities are among the nine #2676/#2748 outer-curvature invariance checks
listed above.

| Historical test identity | Live entry point and executed contract |
| --- | --- |
| `nb_dispersion_is_fitted_theta_hat_not_seed_1463` | `family_noise_parameter` with a struct-literal NB spec (`negative_binomial_log` was removed): estimated metadata yields the fitted theta, not the seed or the residual scale. |
| `nb_dispersion_honors_user_fixed_theta_1463` | The same picker on a held-fixed spec returns the user theta verbatim and refuses estimated-theta metadata as inconsistent. |
| `nb_dispersion_refuses_unfitted_scale_metadata_1463` | `Unspecified` scale metadata refuses with an unresolved-dispersion error. |
| `curvature_evidence_serializes_as_the_legacy_optional_bool_2561` | `CurvatureEvidence`, and the certificate that publishes it under `hessian_psd`, serialize as null, true or false; every unmeasured state, including the later `CriterionContradicted`, reloads as `NotAvailable`. |
| `an_unmeasured_curvature_is_not_the_same_answer_as_an_admissible_one_2561` | `OuterCriterionCertificate::curvature_verdict` reports unmeasured evidence as `Unevaluated`, never `Admissible`, and does not refuse it. `was_measured` was removed; `psd()` carries the check. |
| `non_pd_schur_predicate_preserves_the_two_substring_conjunct_2598` | The value verdicts of `ArrowSchurError::is_non_pd_schur_complement` for all six variants. The surviving rendered-reader test pins the reader to the predicate, so it cannot see both drift together. |
| `shared_block_diagonal_survives_dense_workspace_reclamation_2548` | `shared_block_diagonal` follows the installed operator on an empty-`hbb` system built with `new_with_per_row_dims_empty_hbb_and_htbeta_cols`. |
| `build_dense_schur_direct_refuses_oversize_border_1017` | Both dense Schur builders refuse an 11.9 GiB border against the fixed 8 GiB budget before any allocation. |
| `proportional_penalties_yield_the_exact_lambda_null_direction_2676` | `PenaltyMapInvariance` certifies one invariance along `(c, 0, -1)`, read through `theta_directions` at unit lambdas (`lambda_basis` was removed). |
| `a_three_term_redundancy_no_pair_can_see_is_certified_2676` | A pairwise-invisible `A_2 = A_0 + A_1` certifies one invariance along `(1, 1, -1)/sqrt(3)`. |

All ten historical bodies are byte-identical at `d484a091a^` and `c0a21b554^`. On MSI
lane sw1, tree `40258a5e8` plus the ten pins, job 384866: `cargo test -p gam-solve --lib`
with the seven gam-solve filters gave **7 passed, 0 failed, 1708 filtered out**;
`cargo test -p gam --test glm owed_1463` gave **4 passed, 0 failed, 60 filtered out**,
the three restored pins beside the one that survived; `cargo check --workspace --exclude
gam-pyffi --all-targets` finished clean and the ban scanner passed. Log:
`/scratch.global/sauer354/sw1-logs/verify.384866.log`.

Mutation controls in job 387027 reproduce each named defect and revert it before
the next step. Forcing the NB picker back to the construction seed turns
`nb_dispersion_honors_user_fixed_theta_1463` and
`nb_dispersion_is_fitted_theta_hat_not_seed_1463` red (2 passed, 2 failed). Deleting
the `CurvatureEvidence` serde attribute, and letting `is_non_pd_schur_complement` and
its rendered reader drift together to a bare `"not positive definite"` match, turns
`curvature_evidence_serializes_as_the_legacy_optional_bool_2561` and
`non_pd_schur_predicate_preserves_the_two_substring_conjunct_2598` red (2 passed,
2 failed). Under that joint drift the surviving
`rendered_verdict_matches_the_value_verdict_for_every_variant_2598` stays green, which
is why the value-verdict pin had to come back. Log:
`/scratch.global/sauer354/sw1-logs/mutate.387027.log`.

## Restored single-source checks: #932 and #2647, September 11

Six tests that `c0a21b554` removed are restored from `c0a21b554^`. Each one
exercises production code on main, and each was run on MSI at a sha containing
its landing commit. The restorations came out of a census of the 23 absent
`*_932` pins: for each pin, whether its subject is still production code with
no remaining test reference. Every helper those pins called was test-local, so
a list of missing helper names says nothing about whether a pin can be restored.

| Historical test identity | Recovery evidence |
| --- | --- |
| `joint_penalized_hessian_is_nonsingular_where_the_likelihood_alone_is_not_2647` | Restored in `crates/gam-models/src/gamlss/tests_2647_gauge.rs` by `6c1cfb4ae`. MSI job 393230 at `6c1cfb4ae` ran it with the next two tests: **3 passed, 0 failed, 0 ignored, 1541 filtered out**, 1.37 s. |
| `binomial_location_scalewiggle_termswith_matern_spatial_blocks_fit_finitely` | Restored in `crates/gam-models/src/gamlss/tests_wiggle_ls.rs` by `6c1cfb4ae`. It is the reproducer that the wiggle family's opt-out from full-span Firth/Jeffreys cites. Same job. |
| `binomial_location_scalewiggle_optimum_is_budget_independent_2647` | Restored in `tests_wiggle_ls.rs` by `6c1cfb4ae`. Same job. |
| `binomial_wiggle_joint_hessian_reduces_to_nonwiggle_at_zero_betaw_932` | Restored in `tests_wiggle_ls.rs` by `b49097172`. MSI job 400179 at `ec8e01cfb`, which contains `b49097172` and the `information_third.rs` compile repair, ran it with the next test: **2 passed, 0 failed, 0 ignored, 1545 filtered out**, 0.18 s. |
| `release_measure_binomial_q_tower3_prune_vs_tower4_932` | Restored in `crates/gam-models/src/gamlss/binomial_q_derivs.rs` by `b49097172`. Its doc now says the prune is not measurably slower, matching the `not_slower` cell it records. Same job: the bit-identity arm ran, and the timing cell runs only in release. |
| `softmax_beta_border_gate_derivative_matches_quad_and_fd_across_tails_932` | Restored in `crates/gam-sae/src/row_jet_program.rs` by `6dd1aa4a4`, together with its test-local `SaeOrder2RowProgramSource` impl, with the fixture nested inside the test. It is the only test on main that references `SoftmaxMoment::gate_first`. MSI job 400815 at `6dd1aa4a4`: **1 passed, 0 failed**, 576 comparisons, maximum f64 condition fraction 4.563e-1, maximum quad five-point condition fraction 3.750e-3. |

Not restored, with the reason each stays absent:

- `release_measure_binomial_q_order4_faa_top_vs_strongest_hand_932`: its
  "production" arm calls a test-only projection inside `mod oracle_tests`, so
  the gate would time test code.
- `nested_dual2_channels_from_channels_roundtrip_932`: recorded under Nested
  derivative algebra above.
- `compiled_graph_schedule_matches_all_backends_every_width_932`,
  `dynamic_schedule_boundary_k14_matches_strongest_hand_vgh_932` and
  `runtime_vector_row_program_matches_strongest_hand_mixed_score_vgh_932`: their
  subject was the test-local `rigid_vector_row_nll`, and production runs
  through `rigid_feature_runtime_nll`.
- The five `bms/cell_moment_assembly.rs` pins: the `empirical_rigid_*_closed_form`
  production functions keep 3 to 6 test references each, and the canonical-flex
  pins compared test-local witnesses.
- The four `bms/gpu/row.rs` pins and `measure_device_vgh_end_to_end_932`: they
  are device-only and cannot run on CPU lanes.
- The two `flex_jet.rs` pins and the other three `row_jet_program.rs` pins:
  their subjects either still have test references or have no production caller.

## Retired: pins whose subject was later deleted or whose contract was replaced, September 12

The census at `7ad913f69` (MSI job 505822,
`/scratch.global/sauer354/pool/restore2818/census-absent.505822.log`) found 224
issue-suffixed `#[test]` names that no `.rs` file declares. This section covers
the names that cannot come back as written, for one of three reasons:

- a later commit deleted the production code the pin exercised;
- a later fix replaced the contract the pin asserted;
- the sweep deleted a production item that nothing outside tests called.

Each group names its commit. Every deleted subject named below was checked with
`git grep -w` over `*.rs` at origin/main on 2026-09-12. Any remaining match is
prose, and that is stated where it occurs. None of these tests is counted as
recovered.

### Stagewise birth engine, deleted by `e6fd4251e`

`e6fd4251e` deleted the stagewise engine under the directive to delete code only
tests use: `StagewiseConfig`, `residual_principal_birth_candidate`,
`top_factor_birth_decoder`, `activity_of`, `BirthCandidateDecision`,
`fit_single_atom_response_in_place` and `column_signal_rank`. The same commit
deleted `sae_intrinsic_seed_initial_coords` and `intrinsic_chart_embedding_axes`.
None of these names appears on main.

- `anchor_scored_birth_prefers_uncontested_factor_2080`
- `residual_principal_fallback_fires_on_disjoint_not_noise_2080`
- `residual_principal_seeds_circle_as_rank2_not_dc_2101`
- `born_circle_survives_on_incumbent_sparse_rows_2109`
- `top_factor_birth_mirrors_circle_seed_2109`
- `certificate_rejects_two_circle_blend_2111`
- `kappa_deflation_extracts_clean_circle_from_dense_torus_2111`
- `dense_torus_fixture_has_2k_signal_dirs_2111`
- `dense_torus_integrated_birth_recovery_2111`
- `serial_birth_ledger_retains_errors_and_selects_the_best_arm_2556`
- `batch_birth_ledger_preserves_harvest_order_and_exact_dispositions_2556`
- `intrinsic_seed_allocates_every_chart_function_2240`: `ae42b188c` restored it,
  and `e6fd4251e` then deleted both the test and its subject.

### #2576 reduced-Schur instruments, deleted by `0b6aac49b`

`0b6aac49b` deleted `resident_schur_elimination_diagonal` and
`reduced_schur_logdet_preconditioner_study`, which only tests reached.

- `elimination_share_of_the_border_diagonal_is_fixture_dependent_2576`
- `resident_schur_elimination_diagonal_matches_operator_diagonal_2576`

### Coefficient-group realizer and diagnostic evaluators, deleted by `48f48f910`

`48f48f910` deleted `realize_coefficient_groups_for_custom_family` (only its own
tests called it) and `evaluate_labeled_outer_criterion_for_diagnostics`, the only
subject of `multinomial_jeffreys_outer_gradient_fd_2612.rs`. The realizer's name
survives only in the module doc of
`tests/misc/misc/composed_config_depth3_layout_consistency_2315.rs`, whose tests
are gone.

- `coefficient_group_labels_cannot_reclassify_base_penalties_2315`
- `tied_and_fixed_base_penalties_use_optimizer_coordinate_priors_2315`
- `composed_depth3_group_priors_land_on_their_own_outer_coordinate_2315`
- `composed_two_of_everything_depth3_layout_stays_consistent_2315`
- `derived_penalty_outer_index_equals_emitted_position_across_spec_zoo_2315`
- `unbiased_outer_gradient_matches_central_differences_2612`
- `jeffreys_armed_outer_gradient_matches_central_differences_2612`
- `the_jeffreys_term_is_live_on_this_fixture_2612`
- `unbiased_outer_hessian_matches_the_gradient_jacobian_2612`
- `jeffreys_armed_outer_hessian_agrees_on_the_curvature_verdict_2612`

### Hand-supplied rho boxes, deleted by `14e1ce6d8` and kept deleted by `87c355b12`

`14e1ce6d8` deleted `RhoBox`, `RhoLowerWall`, `RhoCeiling`, `upper_bounds_for`
and `effective_df_floor_rho_upper_bounds`, together with
`effective_df_floor_box_2370.rs`. The per-file repair `87c355b12` kept "derived
rho domains replacing the RhoBox hand boxes", and SPEC forbids hand-supplied
search boxes. `EFFECTIVE_DF_CEILING`, `EFFECTIVE_DF_FLOOR_RELATIVE_FRACTION` and
`effective_df_floor_rho_upper_bounds` survive only in comments.

- `a_caller_box_that_is_already_inverted_is_a_typed_error_2370`
- `a_pinned_box_yields_a_well_ordered_single_point_box_2370`
- `crossing_between_neg_ceiling_and_the_box_floor_keeps_the_ceiling_2370`
- `derived_upper_bound_never_inverts_the_box_across_the_crossing_range_2370`
- `interior_crossing_still_tightens_the_upper_bound_2370`
- `the_rho_box_constructor_rejects_an_inverted_pair_but_accepts_a_pinned_one_2370`
- `effective_df_ceiling_never_emits_upper_below_true_rho_lower_wall_2370`
- `the_rank_one_relative_floor_is_a_logit_of_the_prior_to_data_odds_2615`
- `the_relative_floor_is_scale_free_so_no_single_fixture_can_call_it_inert_2612`

### Production finite-difference curvature ladder, deleted by `c9481ef60`

`c9481ef60` deleted the second-difference ladder the optimizer used to adjudicate
negative curvature, because SPEC keeps finite differences in tests. The only
surviving `hessian_error_2norm` is a parameter name of
`CurvatureResolution::analytic_weyl` in gam-linalg.

- `an_honest_negative_curvature_escapes_on_the_measured_floor_the_declared_one_hid_2748`
- `the_ladder_recovers_a_criterion_curvature_the_analytic_hessian_got_wrong_2748`
- `the_ladder_spans_signal_and_plateau_on_both_sides_of_the_derived_end_2748`

### Contracts replaced by later fixes

| Historical test identity | Removing commit | What replaced the contract |
| --- | --- | --- |
| `target_dose_only_rescales_the_unit_chord_and_cannot_correct_a_displacement_2263` | `ddba34412` | A target dose now moves the atom's coordinate along its chart and solves `t_to`, so the chord-amplitude behavior this pin described is gone. |
| `bernoulli_marginal_slope_ctn_stage1_recipe_only_dispatches_to_bms_issue_2139` | `9c0e6b484` (#2886) | CTN composition moved into the shared native fit service. The commit deleted the unreachable CTN placeholders and this test with them. |
| `co_routed_frame_sweep_is_fixed_code_descent_2634` | `33dc78655` (#2825) | The tied projector objective is optimized in one-shot block sweeps. The test went in that commit. |
| `block_sparse_open_fixed_point_returns_open_certificate_2275` | `8aa65d500` (#2825, #2275) | The block support step now descends the objective the frame and gamma steps descend, so the K ≫ rank block entry certifies. The pin's assertion is inverted and it is renamed `block_sparse_fixed_point_certifies_once_the_support_step_descends_2275_2825` in `crates/gam-sae/src/tiered/fit.rs`. |
| `tiered_returns_best_effort_open_certificate_at_k_gg_rank_2275` | `8aa65d500` (#2825, #2275) | The same fix makes the K ≫ rank tiered fit certify: its frame residual closes to 1.1e-7 against the unchanged 1e-6 tolerance. The pin's assertion is inverted and it is renamed `tiered_certifies_at_k_gg_rank_once_the_support_step_descends_2275_2825` in the same file. |
| `support_parameter_certificate_survives_raw_global_refusal_2634` | `eb27d1b23` (#2933 F08) | The exact Newton displacement `SaeSupportNewtonDisplacement` now certifies the support fixed point. The componentwise diagonal-scaled check this pin exercised became `first_order_screen`, a schedule that only decides when the certificate is priced. The pin keeps its fixture and intensivity assertions against that screen and is renamed `support_first_order_screen_survives_raw_global_refusal_2634` in `crates/gam-sae/src/manifold/support_term.rs`. |
| `contracting_kkt_tail_remains_eligible_after_eight_plateaus_2653` | `4735a7d2a` | The #2653 frontier certificate (`StallPolishProgressCertificate`, `permits_continuation`) is deleted. The polish is permitted at every armed plateau. |
| `either_kkt_currency_can_pay_but_a_repeated_frontier_cannot_2653` | `4735a7d2a` | As above. |
| `terminal_polish_never_raises_the_kkt_residual_2762` | `7a38b3b2e`, kept removed by `87c355b12` | `14e1ce6d8` resurrected it and `87c355b12` removed it again. `4735a7d2a` deleted the #2762 trajectory stop it guarded. |
| `a_replayed_seed_refusal_never_drops_the_reseed_point_2569` | `934d60aa9`, kept removed by `87c355b12` | `14e1ce6d8` resurrected it and `87c355b12` removed it again. |
| `cross_row_preconditioner_build_honors_pd_floor_1795` | `846c51bff` | The cross-row matrix-free PCG route could no longer run. It is deleted with `ArrowBlockDiagInverse`. |
| `threshold_gate_coordinate_block_theta_adjoint_matches_finite_difference_2500` | `65c63d9df` (#2668) | The quasi-Laplace complexity keeps the coordinate block. `coordinate_block_logdet_theta_adjoint` is deleted. |
| `newton_friendly_regime_admits_floor_to_1e_minus_9` | `7ff6dcf9a` (#2469) | The Levenberg-Marquardt damping window is derived from unit roundoff, and `adaptive_lm_lambda_hint` is gone. |

### Arrow-routed co-fit, deleted by the sweep

`cofit_linear_via_arrow`, `cofit_composed_via_arrow`, `ArrowCofitConfig`,
`build_linear_cofit_term`, `build_composed_cofit_term` and
`cofit_block_and_curved` were production functions that no production artifact
called. `d484a091a` deleted them, and `cb8dd972c` later removed the doc-only
`cofit_arrow.rs` that remained. The 2026-09-11 directive is to delete code only
tests use, so these pins are retired. #2023 is still open, and its owner decides
whether the arrow route returns as production code.

- `arrow_linear_cofit_second_pass_is_a_noop_2023`
- `arrow_routed_linear_tier_matches_or_beats_block_reconstruction_2023`
- `composed_arrow_matches_or_beats_block_cofit_2023`
- `composed_arrow_second_pass_is_a_noop_on_curved_and_framed_tiers_2397`
- `insufficient_iterations_return_error_instead_of_open_arrow_cofit_2023`, which calls
  `cofit_composed_via_arrow`
- `insufficient_rounds_return_error_instead_of_an_open_cofit_2023`, which calls
  `cofit_block_and_curved`

### #932 pins counted by file in the September 11 section

These pins keep the reasons given there. They are named here because that section
counts them by file.

- The five `bms/cell_moment_assembly.rs` pins:
  `canonical_flex_row_program_order2_matches_production_lowering_932`,
  `canonical_flex_row_program_order2_matches_tower_and_scalar_932`,
  `empirical_rigid_kernel_matches_exact_implicit_solve_tower_932`,
  `flex_factored_matches_jet2_degenerate_grids_932` and
  `planted_833_style_omission_is_caught_by_exact_tower_932`.
- The four device-only `bms/gpu/row.rs` pins:
  `bms_flex_row_dense_hvp_materialization_matches_cpu_above_block_cap_932`,
  `generated_cuda_row_kernel_r33_matches_canonical_cpu_lowering_932`,
  `mandatory_required_gpu_workspace_consumes_device_cache_end_to_end_932` and
  `release_measure_generated_bms_full_row_vs_strongest_cpu_932`. The same file's
  `generated_cuda_row_kernel_matches_canonical_cpu_lowering_415` is device-only
  for the same reason.
- The two `flex_jet.rs` pins:
  `cell_moment_recurrence_jet_value_matches_numeric_932` and
  `flex_timepoint_inputs_nested_dual_matches_jet4_contraction_932`.
- The three `row_jet_program.rs` pins:
  `compiled_softmax_schedule_matches_generic_tower_all_channels_932`,
  `independent_compiled_schedule_matches_fixed_oracle_above_old_arity_ceiling_932`
  and `jet_hoist_and_order1_border_beat_redundant_baselines_932`.

### Pins whose production subject was deleted as code only tests used

Each pin below exercised a production item that only tests called. The sweep or a
later deletion commit removed that item, under the 2026-09-11 directive to delete
code only tests use. No such item is named in any `.rs` file on origin/main. Each
group gives the removing commit, found with `git log -S` on the defining file.

| Deleted subject | Removing commit | Retired pins |
| --- | --- | --- |
| `logit_posterior_mean_exact`, the Faddeeva exact-logit oracle | `fd8d65e5c` (#2829) | `test_logit_posterior_mean_exact_no_truncation_bias_1459` |
| `with_log_lambda_block` | `e6fd4251e` | `block_efs_step_reaches_gradient_root_2231`, `block_gradient_matches_central_difference_of_cost_2231`, `block_relevance_has_interior_stationary_minimum_2231`, `outer_criterion_prices_block_relevance_2231` |
| `analytic_outer_rho_gradient_components` | `e6fd4251e` | `full_gradient_hessian_channel_set_matches_finite_difference_2253`, `third_order_forward_sensitivity_hessian_matches_finite_difference_2253`, `frozen_state_per_coordinate_channel_fd_audit_2253`, `k1_softmax_active_rho_gradient_matches_directional_fd_2253`, `exact_a_route_gap_is_two_coordinates_with_two_causes_2515`, `complete_outer_gradient_deflation_contribution_is_route_independent_2712` |
| `analytic_outer_rho_gradient_components` plus `schur_inverse_block_deflated` (`e3b50feaf`) | `e6fd4251e` | `zz_measure_smoothness_dof_bundle_vs_deflated_2499` |
| `analytic_outer_rho_gradient_components` plus `coordinate_block_ard_log_precision_hessian_trace` and `coordinate_block_logdet_theta_adjoint` (`65c63d9df`, #2668) | `e6fd4251e` | `zz_measure_exact_a_geometry_bundle_channels_2515` |
| `coordinate_block_assignment_log_strength_hessian_trace`, removed when the complexity kept the coordinate block | `65c63d9df` (#2668) | `exact_a_quotient_value_and_sparse_trace_share_one_classification_2515` |
| `analytic_outer_rho_gradient_at_converged` | `e6fd4251e` | `threshold_gate_analytic_outer_gradient_assembles_2500`, `threshold_gate_outer_gradient_uses_the_modelled_logdet_channels_2500`, `outer_rho_gradient_is_k_dim_and_n_invariant_1033` |
| `assignment_prior_log_strength_hdiag` | `b66a7d04e` | `threshold_gate_sparse_curvature_operator_is_modelled_2500`, `threshold_gate_sparse_operator_is_not_the_raw_prior_on_deflated_rows_2500` |
| `with_ungated` | `b66a7d04e` | `ungated_logit_slot_carries_zero_gradient_and_curvature_1026` |
| `hybrid_collapse_verdict_from_assignments` | `e6fd4251e` | `hybrid_collapse_verdict_accessor_reports_collapsed_slots_2394`, `collapse_verdict_observable_when_reconstruction_is_bit_identical_2394`, `linear_dominance_curved_fit_not_worse_than_linear_on_linear_data_1026` |
| `linear_span_anchor` | `e6fd4251e` | `linear_span_anchor_reaches_pca_ceiling_at_dictionary_rank_1026`, `linear_span_anchor_reaches_pca_ceiling_at_large_dictionary_rank_1026`, `sparse_routing_strictly_underreconstructs_dense_anchor_1026` |
| `derivative_oracle`'s `DerivativeTraceChannel`, `BranchCertificate`, `MajorizerAnchorMode` and `from_arrow_cache`; for the logit-0 localization also `row_psd_majorizer_logit_derivative` (`843e0fc20`) | `e6fd4251e` | `branch_guarded_dual_oracle_pins_live_softmax_channels_2156`, `end_to_end_dual_vs_analytic_logdet_parity_battery_2156_2144`, `sae_logdet_theta_adjoint_logit0_dense_trace_localization_2156` |
| `separation_barrier_value_and_grad_for_test` | `d484a091a` | `separation_barrier_gated_gradient_matches_fd_1625`, `separation_barrier_is_collapse_prevention_not_bandaid_1522`, `zz_measure_separation_force_vs_c2_2253`, `separation_barrier_analytic_gradient_matches_central_fd_1026` |
| `constant_curvature_kernel_kappa_jets` | `d484a091a` | `kernel_kappa_jets_match_central_fd_1404` |
| `fixed_decoder_step_lean_vs_full_1407` | `d484a091a` | `fixed_decoder_lean_step_equals_full_step_1407` |
| `fit_row_metric` | `d484a091a` | `fit_row_metric_one_shot_matches_fit_then_row_metric_2021` |
| `matrix_free_arrow_evidence_log_det` | `d484a091a` | `beta_gauge_quotient_value_inverse_and_gradient_are_orbit_invariant_2022` |
| `reconstruction_energies` (WBIC audit) | `d484a091a` | `rank_charge_deff_is_piecewise_constant_with_monotone_scale_transitions_2099` |
| the in-frame curved route (`InFrameCurvedConfig`, `CurvedRegion`, `fit_inframe_curved_regions`) | `d484a091a` | `inframe_curved_p4096_feasible_where_dense_joint_ooms_2134` |
| `penalized_quasi_laplace_criterion_streaming_exact`, `streaming_exact_arrow_log_det`, `exact_joint_chart_gauge_basis` | `d484a091a` | `zz_measure_dense_vs_streaming_evidence_logdet_terms_2755` |
| `with_fixed_point_certificate` | `d484a091a` | `analytically_refuted_fixed_point_continues_from_checkpoint_with_bfgs_2653` |
| `tail_probability` and `selection_mean` in `smooth_term_lr.rs` | `418c732d2` | `the_null_spectrum_reaches_the_reference_with_a_parametric_term_2672`, `zz_measure_gaussian_reference_against_the_profiled_scale_2672` |
| `weighted_chi_square_sf`, `signed_weighted_chi_square_sf` | `368528959` | `the_two_routes_to_the_null_spectrum_agree_on_real_fits_2672`, `the_two_moment_summary_is_exact_when_shrunk_and_one_signed_otherwise_2672` |
| `cell_third_derivative_boundary_integrand`, `poly_eval_at` | `e95479f25` | `third_order_self_flux_telescopes_but_third_integrand_jumps_at_c2_knot_1454` |
| `jacobian_radial` | `f83e9aeba` | `jacobian_radial_is_stable_through_flat_and_at_d_le_1` |
| the resident-arrow kernel module (`ResidentRowJetHandle`, `ArrowCurvature`, `accumulate_arrow_blocks`) | `2f844874e` | `resident_arrow_blocks_match_materialized_tower_contraction_1017`, `resident_arrow_curvature_channel_is_live_and_beta_block_is_linear_1017`, `resident_arrow_device_matches_host_reduced_blocks_1017`, `resident_arrow_hvp_matches_dense_block_product_1017` |
| `atom_transport_ladder_reports`, `AtomTransportLadderInput` | `95168f488` | `ladder_first_error_is_deterministic_across_dispatch_1017`, `ladder_parallel_matches_sequential_1017` |
| `emulate_certified_encode_batch`, `emulate_certified_encode_row`, `EncodeAtomDevice`, `encode_reconstruction_error` | `fb2a87bc8` | `device_exhaustive_routing_cost_multiplier_2518` |
| `new_with_empty_hbb_and_htbeta_cols` | `e3b50feaf` | `g_matvec_output_owners_are_bit_reproducible_on_device_2535` |

Two more cannot return as written, for different reasons:

- `certified_central_logdet_difference_refuses_floor_clamp_crossing_2398`: its
  subject, `certified_central_logdet_difference`, was finite-difference
  certification scaffolding in `tests_recovery_split_780.rs`, not production code.
  `c0a21b554` deleted it with the tests that used it, so no production contract
  remains for the pin to test.
- `framed_sae_device_matvec_stage_diff_tiny_1551`: its instrument
  `device_matvec_once` lived in the CUDA module and went with the test in
  `c0a21b554`. The pin is device-only and cannot run on CPU lanes.

### The ungated linear tier, deleted by `b66a7d04e`

`b66a7d04e` deleted `with_ungated`, the only API that marked an atom as routed
with `a_k ≡ 1`. These two pins measured the ungated tier itself, so they are
retired:

- `ungated_linear_background_atom_reaches_pca_ceiling_and_converges_1026`
- `ungated_background_resists_sparsity_pressure_gated_degrades_1026`

`sae_outer_objective_never_advertises_finite_difference_curvature_2253` used the
same fixture helper, but its subject was the outer objective's capability
report. It was restored in `3f27cdd74` with the atom left on the default gate.

### Device-gated pins not restored on CPU lanes

Each of these returns before asserting anything when no CUDA device is
available. Restoring them where the pool lanes run would add passes that
exercise nothing, so they stay absent, as the #932 device pins do in the
September 11 section.

- `complete_device_matches_cpu_every_channel_when_admitted_2304`
- `contracted_device_matches_cpu_reduction_when_admitted_2304`
- `contracted_trace_device_matches_cpu_reduction_when_admitted_2304`
- `device_direct_applies_beta_gauge_quotient_at_composed_cofit_shape_2660`
- `moving_ridge_takes_no_host_rebuild_and_matches_independent_2539`
- `sae_direct_mode_device_engages_on_gpu_1551`, whose skip helper
  `device_present_or_record_skip` `c0a21b554` also deleted

### Root-suite pins whose production subject is gone

These pins live in root `tests/` binaries. A whole-file scan of each pin's
d484a091a^ file found a production callee that no `.rs` file on origin/main
defines. Each subject's removing commit was found with `git log -S` on its
defining file.

| Deleted subject | Removing commit | Retired pins |
| --- | --- | --- |
| `analytic_outer_rho_gradient_at_converged` | `e6fd4251e` | `sae_ift_uses_exact_stationarity_jacobian_softmax_high_residual_1418`, `sae_ift_uses_exact_stationarity_jacobian_threshold_gate_high_residual_1418` (`tests/regressions/misc/owed_1418.rs`); `sae_outer_rho_gradient_channel_decomposition_ordered_beta_bernoulli_2087`, `sae_outer_rho_gradient_channel_decomposition_softmax_2087` (`tests/sae/sae/sae_outer_gradient_fd_gate.rs`, whose module is no longer registered) |
| `fit_pair_surface` in `gam-terms` `structure::anova_atom`, now named only in two root-test module docs | `843e0fc20` | `carve_classifies_bound_vs_separable_feature_pairs_975` (`tests/sae/sae/owed_975.rs`, whose module is no longer registered) |
| `weighted_chi_square_sf` | `368528959` | `zz_measure_size_under_candidate_reference_shapes_2672` |

### A pin whose fixture subject was deleted

This section first retired a second pin, `bms_rigid_nonzero_slope_offset_audit_fits_in_time_370`,
because its fixture called `LatentZPolicy::exploratory_fit_weighted` and
`DeviationBlockConfig::triple_penalty_default`. Neither was deleted outright. `7c185e2c5`
replaced `triple_penalty_default`, which was `Self::default()`, with `Default`, and `f2156a78e`
moved `exploratory_fit_weighted` into gam-test-support as
`exploratory_fit_weighted_latent_z_policy`. The pin is restored. Its outer and inner budget
bounds are now read from `BlockwiseFitOptions::default()`, the options the fit runs with,
instead of a literal 60.

| Deleted subject | Removing commit | Retired pin |
| --- | --- | --- |
| `amortized_encode_batch_fast`, `amortized_reconstruct_batch_fast` and `build_data_driven`, the fast encode/decode path that `oos_train_curved` trained through | `728caa9b1` | `curved_warm_start_matches_or_beats_linear_baseline_out_of_sample_2261` |

### Two pins already disposed of by other records

- `latent_log_sigma_curvature_tracks_gradient_fd_scale_ladder_2566`: two
  existing records cover it. The acknowledgement in the deleting commit
  (`git log -p -- docs/source-removal-changes.json`) records the
  #2901 SPEC rule 16 deletion of the print-only #2566 log-sigma curvature ladder,
  together with its only helpers, `richardson_central_difference` and
  `latent_survival_value_fd_authority`. `docs/public-api-2829-disposition.tsv`
  marks its producer, `latent_survival_log_sigma_curvature_certified`, as
  `retired-by-owner:survival`. The survival owner judged the producer superseded
  rather than restoring it.
- `zz_measure_bernoulli_wide_basis_size_versus_n_2672`: it prints the
  `bernoulli/logit, k = 12` size sweep across `n`, and its body asserts nothing.
  This record first retired it because `ingest` calls the deleted
  `tail_probability`. That reason no longer holds: `d06055079` restores `ingest`
  through `tail_probability_with_bound(..).0`, the call the deleted method made.

### A restored pin whose subject was deleted after landing

- `projection_law_beats_the_moment_matched_normal_on_a_two_row_cone_2446`:
  `4b89ebc12` restored it from d484a091a^. `574449c98` (#2829) deleted its
  subject, `constrained_projection_law`, because no product used it, and that
  landed minutes before the restoration. `94fc37dd1` then deleted the pin and its
  two private helpers, `exact_orthant_expectation` and `normal_infeasible_mass`,
  so gam-solve's lib tests compile. It is retired: its subject is gone.

### Two #2515 print-only scans not restored

`c0a21b554` removed both scans with `tests_exact_a_bundle_2515.rs`. Neither body
reaches an assertion: each prints a table and returns. `scripts/assertionless_tests.py`,
which `test-census.yml` runs on every push, refuses a `#[test]` like that, so restoring
either scan would turn that gate red. The #2712 section above already declined to
bring back a print-only experiment as a passing gate.

- `zz_scan_exact_a_admitted_alpha_2515` prints, for nine ARD precisions, the worst
  per-row eigenvalue of `B` and `A`, the clamped-row count, and whether dense and
  streaming exact-`A` evidence factor. Its witness builders, `ExactAWitness2515`,
  `exact_a_witness_2515` and `exact_a_witness_2515_at_alpha`, had no other caller
  among the restored pins, so they stay deleted too.
- `zz_attribute_the_broken_rung_through_production_2515` prints both routes' value
  and gradient across six smoothing strengths. The asserted half of that sweep is
  restored as `forced_streaming_has_a_gradient_wherever_the_dense_route_does_2515`,
  which walks four of those strengths and fails on a streaming refusal or a
  disagreement wherever the dense route ranks the state.

The other ten pins in the census's last undispositioned rows are restored. `170bf96fa`
and `3cfc54e0d` bring back the #2712 trace pair, the #2144, #1625 and #2330 logdet
finite-difference pins and the #2712 row-selected inverse gate. They run on the
finite-difference anchor harness in `tests_recovery_split_780.rs`, rebuilt without the
deleted derivative oracle. `deea7ad64` restores the three #2515 route-parity pins in a
recreated `tests_exact_a_bundle_2515.rs`.

### Eleven census names a strict recount found undispositioned

The section above says the census's last undispositioned rows are restored or
retired. They were not the last. A recount of
`docs/test-census-2818-inventory.json` at origin/main treated a name as disposed of
only when some `.rs` file declares `fn NAME(`, this record names it, or the
acknowledgement in the deleting commit (`git log -p -- docs/source-removal-changes.json`)
records its deletion. Sixteen more names
failed all three tests. Five are restored:

- `5e2877f1a`: `nested_response_moment_rule_reproduces_the_scalar_gaussian_law_2446`,
  `value_lane_prices_at_shared_fixed_point_2228` and
  `a_redundant_penalty_map_still_fits_and_certifies_2676`
- `70a0289d4`: `truncated_response_moments_beat_the_moment_matched_normal_2679`,
  with its wiggle metadata built through the production knot and block builders
- `d06055079`: `gaussian_null_size_is_calibrated_where_the_expansion_is_exact_2672`,
  with the two #939 size contracts that share its scaffold

The two #2023 pins join the arrow-routed co-fit section above, and
`sae_direct_mode_device_engages_on_gpu_1551` joins the device-gated section. The
other eight are retired here. Each missing callee was read from the pin's own
body at `c0a21b554^`, and its removing commit comes from `git log -S` on origin/main.

| Retired pin | Why |
| --- | --- |
| `criterion_lane_gap_is_exactly_the_evidence_logdet_gap_2509` | calls `penalized_quasi_laplace_criterion_streaming_exact` and `streaming_exact_arrow_log_det`, deleted by `d484a091a` |
| `exact_a_route_parity_holds_on_a_deflated_cache_2515`, `exact_a_route_parity_holds_across_a_deflating_rho_ladder_2515` | call `analytic_outer_rho_gradient_components`, last deleted by `e6fd4251e` (`d484a091a` deleted it and `cb8dd972c` restored it), and the harness `full_basis_probe_bundle`, deleted by `c0a21b554` |
| `laplace_value_and_gradient_are_route_invariant_2515` | the same two, plus `coordinate_block_log_det` (`d484a091a`) and the witness builder `exact_a_witness_2515_at_alpha` (`c0a21b554`) |
| `zz_attribute_the_broken_ladder_rung_2515` | calls `analytic_outer_rho_gradient_components`, and its body asserts nothing |
| `zz_attribute_deflated_route_classification_2515` | prints both routes' classifications and asserts nothing |
| `zz_measure_k2_wide_p_inner_trajectory_2080` | a probe that installs a process-wide `log` logger at debug level for every test in its binary. Its one assertion is that the sweep printed a reading. The #2080 gate it localized, `wide_p_outer_reml_terminates_within_probe_budget_2080`, is on main |
| `inner_kkt_gate_is_extensive_while_the_intensive_certificate_exists_2681` | returns as a pass once its fixture converges. Otherwise it asserts that the refused state stays more than ten times over both convergence bounds, which pins a non-convergence: the inverted expected failure that SPEC rule 16 bans. Its accessor `numerical_message` was deleted by `d484a091a` |

The witness builders named in the section above had one more historical caller,
`laplace_value_and_gradient_are_route_invariant_2515`. It is retired here, so they
stay deleted.

### Three restored iso-kappa measurements retired

`05d39b8be` restored seven iso-kappa pins. Three of them are measurements whose own
doc comments say "reports, never fails": they print a finite-difference or rho-part
ladder and assert nothing about their subject. Tonight's retirements hold every
print-only scan to that standard, so these three are deleted, with their
acknowledgement in the deleting commit (`git log -p -- docs/source-removal-changes.json`):

- `zz_measure_iso_kappa_rail_gradient_fd_2425` and
  `zz_measure_iso_kappa_face_saturation_ladder_2425`. The #2425 rail question stays gated
  by `iso_kappa_rail_gradient_matches_fd_at_both_faces_2444` and
  `outer_gradient_at_large_rho_has_a_lambda_infinity_face_2450`.
- `zz_measure_rho_gradient_part_decomposition_binomial_2623`. Its own doc says the channel (D)
  gate belongs in the commit that fixes it.

### A restored #2638 file retired after landing

`a5daa52e4` restored `crates/gam-terms/src/basis/zz_measure_2638_tests.rs` from
`c0a21b554^`. The four tests call `build_duchon_basis_log_kappa_derivatives(data, spec)`,
the psi-jet entry point that resolves a cold spec's chart before differentiating.
`dc325f190` deleted that entry point as unreferenced, and gam-terms' lib tests stopped
compiling. The surviving `build_duchon_basis_log_kappa_derivativeswith_collocationwithworkspace`
takes explicit centers and transform, so the cold-spec contract the file pins has no
production subject. The body is removed again, and its identities are recorded in
the deleting commit's acknowledgement (`git log -p -- docs/source-removal-changes.json`).

`duchon_resolve_chart` is gone as well. `59d30a5b8` (#2829) deleted it with
`ResolvedDuchonChart` and the empty module, because nothing calls it. Before that landing, MSI
job 604886 ran the two asserting pins over `8e9e48c49`, rebuilt on the resolver plus
`build_duchon_basis_log_kappa_derivativeswith_collocationwithworkspace` fed the collocation
points the forward records:

- `duchon_cold_spec_psi_jet_matches_fd_at_the_resolved_chart` passed. The largest
  penalty-block residual was 4.8e-4, on the `constrained` fixture's OperatorMass block (norm
  1.68e2).
- `duchon_resolve_chart_reproduces_the_cold_build` failed on the 2-D `aniso` fixture: the
  resolver adopted a different data-metric reparam `V` than the cold forward build.

That disagreement lives in a function with no product caller, and the function is now
deleted, so both pins stay retired. The forward build still makes these chart decisions itself,
through `duchon_resolve_radial_chart` and `spatial_identifiability_transform_from_design`.

### Two restored root files that did not compile at `9c266da56`

The tip check at `9c266da56` failed on the `inference` and `perf_scale` binaries, with
10 errors in each of two files restored from `c0a21b554^`:

- `tests/inference/misc/conformal_coverage_quality.rs` (`a5daa52e4`):
  `conformal_calibrator_pure_math_matches_split_conformal_definition` and
  `conformal_is_honest_about_too_small_calibration_set` call
  `ConformalCalibrator::from_residuals_and_scales`, `calibrated_interval` and
  `ResponseBounds::UNBOUNDED`, which are `pub(crate)` in gam-predict. They also call
  `certifies_finite()` and `q_hat()`, which no longer exist. The first test is removed:
  `multiplier_is_exact_order_statistic` and `multiplier_does_not_interpolate` in
  `crates/gam-predict/src/conformal.rs` pin the order statistic, and the file's two
  end-to-end arms pin realized coverage through `predict_full_uncertainty_conformal`. The
  second test moves into `conformal.rs`'s test module and reads the `q_hat` field. Both
  root identities are recorded in the deleting commit's acknowledgement
  (`git log -p -- docs/source-removal-changes.json`).
- `tests/perf_scale/misc/row_metric_loud_vs_loadbearing.rs` (`358e2a197`):
  `AtomLensEntry::is_represented_not_used` and `is_used` were deleted by `d484a091a`. They
  read `discrepancy >= 0.5` and `discrepancy <= 0.0`. The test already asserts
  `loud_disc > 0.5` and `quiet_disc <= 0.0` on the live field, so the flag assertions are
  dropped and the cross-over is asserted on the two discrepancies.

### Unsuffixed swept tests: files cut to their module doc

The census covers issue-suffixed names only. A scan at origin/main found the other half
of the sweep: 56 test files that `c0a21b554` cut to their module doc. Their tests have no
issue suffix, so neither the census nor this record saw them. Restored so far:

- `8ade9c77f` (`fbcb5d8c8` then dropped two deleted option fields): the five
  `tests/regressions` files for #682, #582, #584 and the two PIRLS convergence guards
- `a5daa52e4`, `31fa35d5b` and `8a850262e`: `tests_joint_vs_cascade_2131.rs`,
  `latent_coord_design_jacobian_frame_fd_2643.rs`, `duchon_lazy_anisotropic_reparam_1818.rs`,
  `owed_1448.rs`, `conformal_coverage_quality.rs` and `tests_deflation_traces_780.rs`.
  `5d74ecb1b` (#2899) later deleted `tests_joint_vs_cascade_2131.rs`. Its tests
  `split_single_circle_is_a_lower_tail_gap`, `gated_torus_fires_scale_invariant` and
  `phase_correlation_is_invisible_to_energy_screen` went with the pairwise energy screen and
  conditionality fit they exercised, and the acknowledgement in the deleting commit
  (`git log -p -- docs/source-removal-changes.json`) records that energy-screen deletion
- `497f37257`: the Beta and Tweedie arms of the dispersion location-scale variance gate
- `358e2a197`: `row_metric_loud_vs_loadbearing.rs` (`from_blocks_with_mode` is the
  `_and_manifolds` form on Euclidean blocks) and the #1124 negative-binomial seed-spec test

`07a6cfb7e` deleted `gaussian_reml_weight_rescaling_changes_fit.rs` and its test
`gaussian_reml_fit_is_invariant_to_global_weight_rescaling` as an expected-red
module under SPEC rule 16. `c0a21b554` left each file below at its module doc. For each
one, a production function its tests call is gone from origin/main, or the tests exercised
a test-only harness that `c0a21b554` removed with them. The removing commit is the one
that deleted the function's declaration. The comment-only files are now deleted with their
`mod` lines, together with `tests/regressions/misc/owed_1418.rs` (retired in the root-suite
section above) and three doc-only modules that describe code moved or removed long before
the sweep: `crates/gam-gpu/src/kernels/mod.rs`, `crates/gam-terms/src/smooth/tests.rs` and
`tests/perf_scale/misc/large_scale_margslope_repro.rs`. When `tests/perf_scale/sae/` and
`tests/prediction/gpu/` lose their only module, their `mod.rs` files and parent `mod` lines
go too.

| File | Tests | Why they stay absent |
| --- | --- | --- |
| `crates/gam-models/src/gamlss/tests_outer_derivatives.rs` | `outer_lamlgradient_matches_finite_differencewhen_joint_exact_path_is_active`, `rho_only_outer_objective_matches_joint_hyper_when_psi_is_empty`, `outer_lamlgradient_diagonal_binomial_location_scale_matchesfd`, `outer_lamlgradient_diagonal_binomial_location_scale_hard_case_matchesfd`, `outer_lamlhessian_joint_exact_binomial_location_scale_matchesfd`, `outer_lamlhessian_joint_exact_binomial_location_scale_hard_case_matchesfd` | calls `evaluate_rho_outer_criterion_for_diagnostics`, deleted by `48f48f910` |
| `crates/gam-models/src/gpu_kernels/cubic_cell/host_substrate.rs` | `host_oracle_accepts_empty_workload`, `host_oracle_rejects_unsupported_degree`, `host_substrate_matches_cpu_for_quartic_finite_cell`, `host_substrate_matches_cpu_for_sextic_finite_cell_at_d21`, `host_substrate_matches_cpu_for_affine_tail_cell`, `host_substrate_matches_cpu_for_whole_line_affine`, `host_substrate_zeros_invalid_cell_and_records_status`, `cubic_cell_substrate_parity_against_cpu_evaluator` | calls `validate_host_view`, deleted by `d484a091a` |
| `crates/gam-sae/src/manifold/probe_report_cost_2757_tests.rs` | `probe_2757_report_phase_profile_euclidean`, `probe_2757_report_phase_profile_gauge_driving`, `probe_2757_gauge_branch_cost_law` | test-only harness `unit_rho_for_probe` (`tests_frame_curvature_2757.rs`), removed with them by `c0a21b554`; no production subject |
| `crates/gam-sae/src/manifold/tests_crosscoder_rho_2231.rs` | `outer_criterion_prices_block_relevance_2231`, `block_relevance_has_interior_stationary_minimum_2231`, `block_gradient_matches_central_difference_of_cost_2231`, `block_efs_step_reaches_gradient_root_2231` | calls `with_log_lambda_block`, deleted by `e6fd4251e` |
| `crates/gam-sae/src/manifold/tests_graph_atom.rs` | `graph_atom_reads_continuous_circle_as_one_loop_from_knn_edges`, `graph_atom_reads_weekdays_as_atomic_cycle_without_fixed_menu_selection`, `learned_graph_reads_path_as_interval`, `learned_graph_reads_two_disconnected_cycles`, `non_uniform_cycle_reports_no_standard_name`, `learned_graph_reads_branching_tree_and_detects_branch_vertex`, `two_date_modular_synthetic_binds_super_resolution_to_graph_base`, `coactivation_ring_enrolls_as_graph_atom_with_cycle_betti` | calls `knn_candidate_edges`, deleted by `fea430c0c` |
| `crates/gam-sae/src/manifold/tests_graph_spectral_decode.rs` | `eigengap_selects_two_for_circle_and_decouples_from_betti`, `spectral_penalty_is_the_graph_dirichlet_form`, `nystrom_recovers_noisy_circle_angle`, `nystrom_jet_matches_central_difference`, `spectral_decode_beats_single_circle_on_figure_eight`, `nystrom_evaluator_matches_batched_coordinates` | calls `spectral_decode_basis`, deleted by `fea430c0c` |
| `crates/gam-sae/src/manifold/tests_tier0_shared_mean_2023.rs` | `tier0_fit_demeans_and_reconstruction_adds_mean_back`, `tier0_makes_dc_zombie_ev_invisible_six_circles` | calls `fit_tier0_mean`, deleted by `e6fd4251e` |
| `crates/gam-solve/src/reml/boundary_laml.rs` | `log_boundary_g_matches_direct_integral`, `log_boundary_g_zero_multiplier_is_half_gaussian`, `log_boundary_g_large_positive_ratio_is_reciprocal`, `log_boundary_g_far_interior_recovers_gaussian`, `log_boundary_g_derivatives_match_central_difference`, `interior_factor_joins_active_factor_at_the_boundary`, `interior_factor_far_from_boundary_recovers_full_gaussian`, `interior_factor_matches_gaussian_tail_integral`, `interior_factor_derivatives_match_central_difference`, `log_gaussian_orthant_diagonal_is_exact_product`, `log_gaussian_orthant_first_order_correction_beats_product`, `log_gaussian_orthant_bracket_contains_exact`, `log_gaussian_orthant_bracket_escalates_on_nonpositive_curvature` | calls `log_boundary_g`, deleted by `d484a091a` |
| `crates/gam-solve/tests/suite/sae_evidence_matvec_1017.rs` | `evidence_matvec_deterministic_and_matches_cpu`, `evidence_matvec_utilization_loop` | calls `sae_framed_schur_matvec_cpu`, deleted by `aa3e5cf99` |
| `tests/arrow_gpu/gpu/gpu_numerical_stability.rs` | `pirls_gpu_matches_cpu_across_stability_grid`, `reml_gpu_logdet_and_score_match_cpu` | test-only harness `gpu_gate` (`tests/common/gpu/gpu_gate.rs`), removed with them by `c0a21b554`; no production subject |
| `tests/autodiff/optimization/channel_hessian_matches_fd.rs` | `bernoulli_channel_hessian_matches_fd` | calls `from_eta_pilot`, deleted by `d484a091a` |
| `tests/basis_smooth/smooths/bspline_derivative_fd_oracle.rs` | `bspline_derivatives_1_through_4_match_central_finite_differences`, `bspline_derivative_matches_fd_on_uniform_open_knots`, `bspline_derivative_partition_of_unity_sums_to_zero` | calls `evaluate_bsplinesecond_derivative_scalar`, deleted by `d484a091a` |
| `tests/glm/misc/beta_generative_phi_drives_draw_variance.rs` | `beta_generative_draw_variance_tracks_forwarded_phi_not_seed` | calls `sampleobservation_replicates`, deleted by `d484a091a` |
| `tests/identifiability/misc/topology_mixture_refinement.rs` | `off_ladder_truths_k4_and_k6_are_recovered_exactly`, `in_ladder_truth_k7_is_unaffected_and_bracketed`, `circle_truth_refinement_brackets_instead_of_creeping` | calls `fit_mixture_rung`, deleted by `b6bda0923` |
| `tests/identifiability/misc/topology_race_calibration.rs` | `repeated_draws_are_accurate_and_decisive_calls_are_never_wrong` | calls `fit_mixture_rung`, deleted by `b6bda0923` |
| `tests/identifiability/misc/topology_two_verdict_race.rs` | `circle_read_discretely_yields_two_different_verdicts`, `quadrant_readout_computational_verdict_recovers_k4` | calls `fit_mixture_rung`, deleted by `b6bda0923` |
| `tests/identifiability/misc/topology_union_candidates.rs` | `two_circles_prefer_structured_union_over_single_torus_and_circle`, `circle_plus_outlier_cluster_prefers_structured_union_over_pure_rungs`, `single_circle_negative_control_does_not_prefer_any_union`, `fit_union_candidate_prices_by_total_parameter_count` | calls `fit_union_rung`, deleted by `b6bda0923` |
| `tests/measure_jet/misc/measure_jet_ell_outer_gradient_fd_2761.rs` | `measure_jet_ell_outer_gradient_matches_fd_without_double_penalty`, `measure_jet_ell_outer_gradient_matches_fd` | calls `enable_outer_gradient_fd_capture`, deleted by `1bc46ac50` |
| `tests/misc/misc/composed_config_depth3_layout_consistency_2315.rs` | `composed_two_of_everything_depth3_layout_stays_consistent_2315`, `composed_depth3_group_priors_land_on_their_own_outer_coordinate_2315` | calls `realize_coefficient_groups_for_custom_family`, deleted by `48f48f910` |
| `tests/perf_scale/misc/pair_surface_grid_consumer.rs` | `pair_surface_grid_backend_matches_dense_oracle`, `pair_surface_feeds_carve_additive_splits_bound_refuses`, `pair_surface_large_gridded_n_recovers_truth_end_to_end` | calls `fit_pair_surface`, deleted by `843e0fc20` |
| `tests/perf_scale/misc/power_law_analyzer.rs` | `fit_recovers_clean_power_law`, `fit_rejects_insufficient_data`, `fit_rejects_degenerate_x_collapse`, `fit_flags_outlier_in_max_log_resid`, `fit_reports_input_length`, `power_law_fit_struct_roundtrips`, `report_extrapolation_verdicts_track_budget_boundary`, `report_returns_fit_but_skips_extrapolation_when_fit_poor`, `fit_ignores_non_positive_or_non_finite_points`, `fit_recovers_random_clean_power_laws` | test-only harness `fit_power_law` (`tests/perf_scale/misc/power_law_common.rs`), removed with them by `c0a21b554`; no production subject |
| `tests/perf_scale/misc/row_measure_enrichment.rs` | `enrichment_oversamples_rare_loud_feature_without_touching_loss`, `no_harvest_is_todays_uniform_behavior` | calls `is_enriched`, deleted by `272905c19` |
| `tests/perf_scale/sae/rho_posterior_tier1_sae_coverage.rs` | `tier1_rho_quadrature_improves_sae_smooth_band_coverage` | calls `rho_posterior_tier1_quadrature`, deleted by `1fdca1867` |
| `tests/perf_scale/smooths/glm_frozen_w_tensor_n_independence.rs` | `glm_frozen_w_outer_objects_are_n_independent`, `glm_frozen_w_accessor_shapes_are_fixed_k_across_n` | calls `d2gram_dpsi2`, deleted by `272905c19` |
| `tests/perf_scale/smooths/grid_spline_2d_exact_oracle.rs` | `streaming_band_assembly_matches_dense_oracle`, `reml_fit_beats_the_noise_floor`, `assembled_penalty_matches_closed_form_quadratic_energy` | calls `fit_grid_spline_2d_at`, deleted by `dc325f190` |
| `tests/prediction/gpu/predict_on_cpu_only_host_does_not_panic_with_cudarc.rs` | `predict_after_load_on_cpu_only_host_does_not_panic_with_cudarc` | calls `cuda_driver_available`, deleted by `5d498d0e9` |
| `tests/quality/families/quality_vs_brute_force_loo_binomial_logit.rs` | `alo_eta_tilde_matches_exact_loo_binomial_logit` | calls `compute_alo_diagnostics_from_fit`, deleted by `272905c19` |
| `tests/quality/families/quality_vs_scipy_sandwich_glm_gaussian.rs` | `gam_alo_sandwich_ci_covers_true_linear_predictor_at_nominal_rate` | calls `compute_alo_diagnostics_from_fit`, deleted by `272905c19` |
| `tests/quality/misc/quality_corrected_aic_psis_loo_selection.rs` | `corrected_aic_penalizes_at_least_as_much_as_conditional`, `psis_loo_paired_comparison_prefers_the_true_generator` | calls `compute_alo_diagnostics_from_fit`, deleted by `272905c19` |
| `tests/quality/misc/quality_mixture_rung_vs_reference.rs` | `cluster_regime_gam_selects_mixture_and_recovers_k_match_or_beat_sklearn`, `circle_regime_gam_selects_smooth_circle_not_mixture_via_interpolated_holdout` | calls `fit_mixture_rung`, deleted by `b6bda0923` |
| `tests/quality/misc/quality_vs_mgcv_pair_surface_live_backend.rs` | `fit_pair_surface_recovers_truth_and_matches_or_beats_mgcv_te` | calls `fit_pair_surface`, deleted by `843e0fc20` |
| `tests/quality/misc/quality_vs_scipy_spd_frechet_mean.rs` | `spd_frechet_mean_is_the_riemannian_center_of_mass`, `spd_frechet_mean_is_the_riemannian_center_of_mass_on_real_data` | calls `spd_frechet_mean`, deleted by `210b5196b` |
| `tests/regressions/survival/cloglog_survival_large_sigma_asymptotic_biased_low.rs` | `cloglog_survival_value_matches_reference_in_large_sigma_band` | calls `log_kernel_term`, deleted by `d484a091a` |
| `tests/survival/survival/survival_marginal_slope_outer_gradient_fd_1040.rs` | `survival_marginal_slope_outer_gradient_fd_audit_matern`, `survival_marginal_slope_outer_gradient_fd_audit_duchon` | calls `enable_outer_gradient_fd_capture`, deleted by `1bc46ac50` |
| `tests/perf_scale/smooths/grid_spline_2d_streaming_bench.rs` | `n_10_000_000_streaming_acceptance_bench` | calls `GridSpline2dDesign`, the 2-D grid spline engine that `85abc4592` retired because no product used it |
| `tests/identifiability/misc/ladder_cert_rate_measure.rs` | `report_non_affine_ladder_cert_distribution_on_flex_path` | a #979 measurement whose one assertion is that cells were evaluated. Its import `gam::families::cubic_cell_kernel` no longer resolves, because the kernel lives in gam-model-kernels and the facade does not re-export it |

### Shells whose callees survive under other names

These files were cut to their module doc, but every production callee their tests need is
still on main, renamed, moved or reshaped. They are restored:

| File | Tests | Adaptation |
| --- | --- | --- |
| `tests/survival/survival/owed_1388.rs` | 3 | `canonicalize_for_identifiability` is now `canonicalize_for_identifiability_with_operating_scalars`. The old function forwarded `None` for the operating scalars, and the tests pass `None`. |
| `tests/inference/misc/margslope_smallcondition_smoke.rs` | 2 | the two constructors moved, as the #370 correction above describes. The module doc no longer cites two large-scale reproducers that are gone. The flex arm is red. At MSI job 604886, its n=2000 fit sat in `joint Newton hessian_qp cycle=9` for more than 10 minutes against its 60 s budget. `88c8280d1` dropped the arm and the next landing restored it, because the stall is a production defect that a test should keep showing. The evidence is posted on #979. |
| `tests/identifiability/misc/constant_curvature_kappa_coverage_sims.rs` | 3 | `f46ec2bb2` deleted the uncalled `KappaEstimateSupport::is_railed`; the tests compare against `KappaEstimateSupport::Interior`. `e1f90bec8` removed the `pilot_subsample_threshold` option, and its line is dropped. |
| `tests/quality/families/quality_vs_pymc_nuts_binomial_logit.rs` | 2 | `NutsConfig` no longer carries `nwarmup` or `n_chains`: gam runs `NUTS_CHAINS` chains and ends warmup when adaptation stabilizes. The PyMC baseline's chain count reads `gam::hmc::NUTS_CHAINS`, and payloads use `MODEL_PAYLOAD_VERSION`. |
| `tests/inference/misc/bms_audit_nonzero_slope_baseline_370.rs` | 1 | the rigid #370 pin, as above |

`tests/autodiff/misc/contract_gradient_gates.rs` (1 test) is restored with six of its seven
rows. `09d533460` replaced `evaluate_externalcost_andridge` with `evaluate_externalcost`, which
returns the cost alone, and `a88c62eee` removed `ExternalOptimOptions`' two Kronecker fields.
The gate now evaluates every row before it asserts, so one red row cannot hide the others.
Its two SAE rows take their central differences in place on the assembled term and restore it
afterwards. `assemble_arrow_schur` freezes the barrier coactivation on the term it assembles,
and the gradient treats that coactivation as constant (#1625). `SaeManifoldTerm::clone` drops
the frozen gates, and `barrier_coactivation_pairs` then recomputes coactivation from live
routing. A difference taken on clones therefore differentiates another objective, and on the
K=2 softmax row that showed up as a flat logit residual.

The `survival/laml-net-single-block` row is not restored. Its subject,
`WorkingModelSurvival::evaluate_survival_lamlcost_and_gradient`, was a test shim that
`d484a091a` deleted together with its private inner-mode reconvergence. The unified survival
LAML evaluator it wrapped, `unified_lamlobjective_and_rhogradient`, is `pub(crate)`, so no
public route re-solves the inner mode at a perturbed ρ.

## Runtime verdict of the restored pins, September 17

By September 16 every census name was declared or recorded. These runs execute the declared
pins on clean checkouts (`HEAD == PIN`, DIRTY=0). Each target gets one `cargo nextest` run, and
its filter selects exactly the pins declared in that target, with the expected count printed.

| run | sha | targets | pins | result |
| --- | --- | ---: | ---: | --- |
| 1100763 | `a24b42fb0` | 16 that do not depend on gam-sae | 71 | 71 passed. The other 15 targets did not compile there (gam-sae E0658 `float_bits_const`). |
| verify3.1109689 | `39e96f017` | the 15 that depend on gam-sae | 123 | 101 passed, 22 not green |
| 1190023 | `0e9ca555de` | gam-sae lib | 44 | 31 passed, 13 not green |
| 1190027 | `0e9ca555de` | the five root pins still not green, run serially | 5 | 2 passed, 3 not green |

Between `39e96f017` and the pin, five pins turned green:
- the #2144 pair and #1625, after `2ef4a4ecc`;
- `value_lane_prices_at_shared_fixed_point_2228`;
- `large_scale_convergence_regression` and `large_scale_dense_logit_regression_guard`. Both
  were red under concurrent nextest at `39e96f017` and pass alone at the pin, in 0.50 s and
  1.38 s. #2668 recorded on 07-31 that the second one's wall-clock bar cannot separate signal
  from load.

Not green at the latest run:

| pin | latest verdict | disposition |
| --- | --- | --- |
| `sae_logdet_theta_adjoint_matches_fd_on_deflated_fixture_2330`, `ard_log_precision_hessian_trace_from_probes_matches_dense_on_deflated_rows_2712`, `assignment_log_strength_hessian_trace_from_probes_matches_dense_on_deflated_rows_2712`, `sae_row_selected_inverse_from_probes_is_the_deflated_block_2712` | 1190023: no member of the declared anchor family certifies `SomeRowDeflates`. On the softmax ladder the joint exact observed information is indefinite at every member; the ordered Beta–Bernoulli ladders never deflate a row. | #2822, which names all four |
| `ard_log_precision_trace_matches_dense_fd_pd_region_deflation` | 1190023: no deflated direction. At `7a12efd48e` (regate 1179983) the closest gauge direction sat 4.5e7 to 9.9e7 times above the bar, so the fixture has no near-null orbit. | #2822 |
| `dense_and_arrow_materialize_the_same_raw_exact_a_2515`, `forced_streaming_has_a_gradient_wherever_the_dense_route_does_2515`, `forced_streaming_admits_a_deflating_state_and_matches_dense_2515` | 1190023: the same anchor-family refusal | #2822, routed to #2267 |
| `zz_planted_circle_plain_engine_stall_diagnostic_2234` | 1190023: 59 infeasible criterion evaluations returned | #2234, and named on #2822 |
| `existence_and_intensity_are_separately_identified_1939` | 1190023: the dead atom's held-out contribution is 0.0350 against the weak live atom's 0.0597 (the bar is 0.25×). It passed at `d0e26faca` (census 616588), and its file and fixture builder are unchanged since `4da8f50c1`. | #2822, comment 5720624932 |
| `sae_outer_objective_never_advertises_finite_difference_curvature_2253`, `two_circle_whitened_k2_recovers_disjoint_signal_2027`, `two_circle_separates_at_narrow_and_wide_widths_2027` | 1190023: refused at entry for an identically zero decoder, which `8f14fd3744` made a refusal | #2822 fix-forward by ad-sae (sae2822-suite), released by i2818 on 09-17 (patch `/scratch.global/sauer354/pool/i2818/fix.patch`, md5 b77d2b15). It seeds each fixture from the data least-squares decoder at its chart and ρ (`refit_decoder_least_squares_at_current_state`), as `8f14fd3744` seeded its siblings. Assertions and bars are unchanged. Pool job 1199628 runs the three pins at `0e9ca555de` without and then with the patch. |
| `corrected_covariance_nodes_are_criterion_calibrated_2728` | 1190027: the cubature correction has trace −7.49e-2, against 6.48e-2 for the first-order correction | the #2627 root census, among t2627-root's singles |
| `flex_full_outer_completes_under_budget_683` | 1190027: TIMEOUT at 600 s, run alone | the inner-solve non-termination recorded on #979 |
| `gaussian_null_size_is_calibrated_where_the_expansion_is_exact_2672` | 1190027: TIMEOUT at 600 s, run alone. The arm is 2 k × 2 n × 120 replicates, and its Poisson and Bernoulli siblings were halved to fit the 300 s slow period; this arm was not. | no owner yet. Pool job 1199425 measures its uncapped wall time at `0e9ca555de`, which decides between a measured nextest override and a production hang. |
| `survival_marginal_slope_follow_up_mode_response_matches_fd_2765` | verify3.1109689: TIMEOUT at 600 s; not re-run at the pin | #2765, owned by i2765 |
| `gam_nuts_binomial_logit_recovers_truth_and_is_calibrated` and its real-data arm | verify3.1109689: `REFERENCE_ENV_MISSING:pymc`; the real-data arm hit TIMEOUT | not measured on MSI, because pymc is not installed there |

Logs: `/scratch.global/sauer354/pool/restore2818/verify.1100763.log`,
`/scratch.global/sauer354/pool/restore2818/verify3.1109689.log`, and
`/scratch.global/sauer354/pool/i2818/tip1.1190023.log` and `tip2.1190027.log`.
