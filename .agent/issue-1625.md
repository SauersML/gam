# Issue #1625 — SAE row-jet oracle convergence

## Status: issue CLOSED on GitHub with 2 fixes on main (c65fc4e, a5f099e)

## Plan
1. Reproduce CURRENT state of the two oracle tests on main.
2. The issue noted ADJACENT failures (4 dense-FD logdet-trace oracles) still red.
3. Verify whether closure is genuine; if tests are still red, fix the true root cause.
4. Expand: harden inner (t,β) Newton convergence on marginal fixtures.

## Tests in question
- manifold::tests_row_jet_and_outer_objective_780::sae_row_jet_program_matches_production_row_jets_on_converged_cache
- manifold::tests_row_jet_and_outer_objective_780::batch4_jet_lanes_match_scalar_hand_row_jets
