# Test integrity and removal evidence

The Rust library, CLI, Python extension, examples, benchmarks, and test targets
are distinct roots of the source graph. Absence from a linked binary's symbol
table is never evidence that an item is unused: generics, inlining, and LTO can
remove symbols for live code. A production reachability sweep must exclude
`#[cfg(test)]` code from its deletion candidates and account for all supported
library APIs and configurations. Test helpers remain test code; moving them to
production or duplicating them inline does not repair a faulty reachability rule.

Before removing code, trace its source references and public API contracts,
identify its callers in every supported target, and compile the affected targets.
Tests whose production behavior was removed need a semantic retirement decision;
tests that exercise surviving behavior need repair. Compilation alone cannot tell
the two apart. Do not delete a failing test merely because its fixture was deleted.

`scripts/test_census.py --base BASE --head HEAD` compares immutable Git trees,
independent of the worktree or index. It reports the Rust-file denominator,
ordinary test declaration count, and issue-suffixed test identities under
`crates/`, `tests/`, and `src/`. Comments and string literals cannot stand in for
a test. Duplicate names retain their multiplicity, and moving a test between
files preserves its identity. Standard `#[test]` and conditional
`#[cfg_attr(..., test)]` declarations are counted irrespective of target/feature
conditions. This is a source census, not an assertion that declarations are
registered, compiled, or executed; compiled test listings and runtime verdicts
are separately required to prove regression coverage.

A decreased total count or missing issue-pinned identity fails the gate. The
comparison's base commit supplies the independent baseline, so deleting both a
test and its documentation cannot make the loss disappear. An empty source,
test, or pin denominator is a failure even with a removal acknowledgement.

For an intentional removal, append one object to `docs/test-census-changes.json`
in the same change, with these exact fields:

```json
{
  "base": "full pre-change commit SHA",
  "test_count_decrease": 1,
  "removed_pins": {"old_test_name_1234": 1},
  "reason": "Describe the behavior removed or the semantic replacement.",
  "evidence": "Name the live replacement tests and their executed results, or link the reviewed retirement decision."
}
```

Counts and identities must match the measured loss exactly; blanket allowances
and stale acknowledgements cannot authorize a new removal. The text is review
evidence, not something a name/count scanner can independently establish. A green
census does not discharge historical missing pins in #2818: those require
individual recovery or recorded retirement, plus actual execution evidence.

The census workflow runs for every main push without cancelling superseded
commits. It also runs its planted-loss and empty-population controls. Its JSON
artifact records both complete source inventories and the comparison subjects.
