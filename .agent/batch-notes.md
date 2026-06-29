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
