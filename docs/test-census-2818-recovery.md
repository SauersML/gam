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

Two other historical #1410 identities compared active entries bit-for-bit to
the deleted dense convenience methods:
`active_softmax_dense_entropy_hessian_entry_matches_dense_block_1410` and
`active_softmax_majorizer_logit_derivative_matches_dense_1410`. Those wrappers
shared the same expressions as the active entries. Their original comparison
cannot establish an independent oracle for the surviving implementation.
Independent HVP/finite-difference replacements still need explicit registration
and execution evidence before these two inventory entries are discharged.

The remaining historical inventory is still open. A successful source census,
or the restoration of these few contracts, does not establish that all 303
historically deleted pinned identities have been recovered or retired.
