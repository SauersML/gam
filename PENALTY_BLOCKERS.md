# PenaltyMatrix Move Blockers

## Status

`PenaltyMatrix` has been moved into `gam-problem` as `gam_problem::PenaltyMatrix`,
with `gam::families::custom_family::PenaltyMatrix` re-exporting that neutral
contract. The old `src/families/custom_family/penalty.rs` module has been
removed.

No commit was created because the required workspace check did not complete
green.

## Verification Blocker

Required check:

```sh
RUSTC_WRAPPER=sccache SCCACHE_DIR=/Users/user/.sccache-shared cargo check --workspace --target-dir target/penalty2-check
```

Observed blockers:

1. In `/Users/user/gam-wt-penalty2`, the worktree disappeared during the first
   check attempt. Cargo failed with missing intermediate files under
   `target/penalty2-check`, then `/Users/user/gam-wt-penalty2` no longer existed.
2. Reapplied the scoped change in `/Users/user/gam` and retried the check with
   serialized jobs:

```sh
RUSTC_WRAPPER=sccache SCCACHE_DIR=/Users/user/.sccache-shared CARGO_BUILD_JOBS=1 cargo check --workspace --target-dir target/penalty2-check
```

3. The serialized check progressed into dependency compilation but repeatedly
   failed before a clean workspace result:
   - `sccache: Compile terminated by signal 9` while compiling `equator-macro`.
   - `sccache: encountered fatal error` while compiling `pest`, failing to
     create `target/penalty2-check/debug/.fingerprint/pest-*/output-lib-pest`.

The last failure occurred before the workspace check could finish, so the tree
was not committed.
