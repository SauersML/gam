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
# Batch triage — closed issues 1498–1622

## Headline finding: HEAD does not build (foundational blocker)
`./build.sh` aborts with **8 ban-scanner violations across 4 rules**, all introduced
by recent #932 perf/oracle commits. Because the build is broken, NONE of the closed
"fixed" issues can actually be verified at HEAD, and several BUILD-BROKEN issues
(#1614, #1589, #1584, ...) describe exactly this recurring failure mode — fixes
addressed instances, not the pattern, so it keeps regressing.

### The 8 violations
1. `survival/marginal_slope/timepoint_exact/flex_jet.rs:4795` — `#[allow(clippy::too_many_arguments)]`
2. `survival/marginal_slope/timepoint_exact/flex_jet.rs:4879` — `#[allow(clippy::too_many_arguments)]`
3. `bms/gradient_paths.rs:2609` — `#[ignore]` test
4. `bms/cell_moment_assembly.rs:4774` — `#[ignore]` test
5. `gam-sae/src/row_jet_program.rs:1480` — `#[ignore]` test
6. `bms/gradient_paths.rs:2608` — `#[test]` without assertions (the ignored bench)
7. `multinomial_reml.rs:935` — `#[cfg(test)]` on a `src/` impl method
8. `multinomial_reml.rs:1061` — `#[cfg(test)]` on a `src/` impl method

## Plan
- Reopen #1614 (its class of defect is live at HEAD again) and fix the build properly.
- Convert the `#[allow(too_many_arguments)]` oracles to struct-arg signatures.
- Convert the three `#[ignore]` timing benches into real asserting tests (or fold the
  timing into an asserting correctness test) — the ban forbids both `#[ignore]` and
  assertion-less tests.
- Move the two `cfg(test)` multinomial oracle methods into the existing
  `#[cfg(test)] mod tests` impl block (the sanctioned location).
- Then sweep the rest of the range for genuinely-improperly-closed bugs.

## #1515 — count-family degenerate-response guard (reopen)

Reopened: gamfit.fit accepts an all-zero Poisson/NB response (saturated η̂=log(0)=−∞)
while the binomial path and the CLI reject the analogous all-zero/all-one case.
Fix: extend ResponseFamily::validate_response_degeneracy (gam-spec, the shared
core guard both fit entry points already call) to reject all-zero count responses.
