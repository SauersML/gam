# Agent gam-closed-1138-1317 — batch notes

## Range: CLOSED issues #1138–#1317 (153 in range)

## BLOCKER (handling first): main is RED for the whole fleet
`./build.sh` aborts with 9 ban-scanner violations:
- 2× `#[allow(clippy::too_many_arguments)]` — survival flex_jet.rs:4795,4879
- 3× `#[ignore]` timing microbenches — bms/gradient_paths.rs, bms/cell_moment_assembly.rs, gam-sae/row_jet_program.rs
- 1× `#[test]` without assertions — gradient_paths.rs:2608
- 2× `#[cfg(test)]` impl methods — multinomial_reml.rs:935,1061
- 1× tracked file >10k lines — manifold/tests.rs (10012)

All from recently-merged #932 WIP commits. Fix plan:
- flex_jet too_many_args: bundle row tuple into a struct, drop the allow.
- ignored timing microbenches: delete (assert nothing; env-dependent; perf claims live in commits).
- multinomial_reml cfg(test) methods: move into a #[cfg(test)] mod.
- manifold/tests.rs >10k: split off a sub-module file.

## Triage findings (1138-1317)
- #1167 CLogLog AFT ratio derivs: VERIFIED FIXED (returns (-t,t,t,t,t)).
- #1156 Skovgaard expected_info: VERIFIED FIXED (u uses expected_info, regression test present).
- #1218 PG gate +½dlog2π sign: VERIFIED FIXED (now -0.5*d_g*log2pi + regression test).
- More triage pending after build is green.
