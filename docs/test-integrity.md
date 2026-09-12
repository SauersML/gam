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
The public-root rule and concrete API decisions are recorded in
[`rust-library-surface.md`](rust-library-surface.md). Exported generic library
functions remain roots even when neither shipped executable instantiates them.
Tests whose production behavior was removed need a semantic retirement decision;
tests that exercise surviving behavior need repair. Compilation alone cannot tell
the two apart. Do not delete a failing test merely because its fixture was deleted.

`scripts/source_removal_guard.py --base BASE --head HEAD` makes that removal
policy executable over immutable Git trees. It guards every removed public item,
every removed test-scoped item (including helpers rather than only `#[test]`
functions), and every private production item with another Rust-source
occurrence, including in a supported test target. This is intentionally conservative: source references survive
inlining, monomorphisation and LTO, while a linked symbol table does not. A truly
unreferenced private item may be removed directly; guarded removals require an
exact, commit-specific semantic record in `docs/source-removal-changes.json`.
The record is a reviewed retirement mechanism, not a search box or an override
that proves reachability.

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

An empty source, test, or pin denominator fails the census. Lost tests are
reported, not refused.

The comparison against the base names the exact identities that went missing,
so deleting both a test and its documentation cannot make the loss disappear
from the report. Its workspace totals let growth in one crate pay for deletion
in another — the shape #2818 actually had — so `unit_test_decreases` reports the
per-unit view of the same step: `crates/<name>` for anything under a crate, one
unit per integration binary under `tests/`, and the top-level `src` suite.

`python scripts/test_census.py --positive-control` re-measures `c0a21b554`, the
commit this census exists for, and requires that it still reports 2,290 lost
tests, 300 lost pinned names and 38 units that lost tests. A lexer or comparison
change that moves those numbers has to move them in `CONTROL` too, in review. The
census workflow runs this control on every push, so a census that has quietly
stopped detecting anything cannot report green.

A green census does not discharge historical missing pins in #2818: those
require individual recovery or retirement, plus actual execution evidence.

The census workflow runs for every main push without cancelling superseded
commits. It also runs its planted-loss and empty-population controls, and the
`c0a21b554` positive control above. Its JSON artifact records both complete
source inventories and the comparison between them.
