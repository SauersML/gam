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

Three other #2280 identities remain unresolved:
`co_collapse_flags_duplicate_charts_2280`,
`co_collapse_thresholds_bracket_the_gate_2280`, and
`co_collapse_spares_healthy_swiss_roll_atlas_2280`. Their
`LocalAtlas::co_collapse_candidates` query was deleted, and no replacement has
been verified. Because #2280 itself remains open, disappearance of that callable
is not evidence of an intentionally retired product requirement.

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
