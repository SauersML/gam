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
adjoint gate remains pending. The existing cold state's two spectrally deflated
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
block-reconstruction result is distinct from the still-pending
deflation-adjoint sensitivity requirement.

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

The remaining historical inventory is still open. A successful source census,
or the restoration of these few contracts, does not establish that all 303
historically deleted pinned identities have been recovered or retired.
