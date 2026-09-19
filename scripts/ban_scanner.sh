#!/usr/bin/env bash
# Run the repository's umbrella ban scanner on its own, in seconds.
#
# The scanner is the root crate's `build.rs`. Cargo only runs it as a side
# effect of building the root crate, which means:
#
#   * `cargo check -p <member-crate>` has ZERO scanner coverage — a lane that
#     verifies its own work per-crate sees green and lands a violation that
#     breaks every root-crate target in the workspace, including the whole
#     quality suite (#2582);
#   * when it does run, its report arrives as build-script stderr under a
#     `failed to run custom build command for 'gam'` headline, several hundred
#     `cargo:rerun-if-changed=` lines away from the file:line it names.
#
# `build.rs` is std-only and reads the tree through `CARGO_MANIFEST_DIR`, so it
# compiles and runs directly. This is the SAME code cargo runs — not a
# re-implementation of its rules — compiled from the tree under test, which is
# what makes "scanner verified" mean anything.
#
# It then runs scripts/spec_ban_ratchet.py, which checks the SPEC bans that live
# in the shape of code (grid search, hand boxes, jitter, unconverged successes,
# magic thresholds, finite differences, GCV, Python-side math) against the
# shrink-only ledger scripts/spec_ban_ledger.tsv.
#
# Usage:  scripts/ban_scanner.sh [repo-root]
# Exit:   0 clean, 1 violations (each printed as `error: <file>:<line>: <src>`),
#         2 either scanner could not run.

set -uo pipefail

root="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
if [ ! -f "${root}/build.rs" ]; then
  echo "ban_scanner: no build.rs at ${root}" >&2
  exit 2
fi

work="$(mktemp -d)"
trap 'rm -rf "${work}"' EXIT
mkdir -p "${work}/out"

# The scanner is compiled from the tree being scanned. A binary built from an
# older checkout applies that checkout's rules and will disagree with the gate.
if ! rustc --edition 2024 -O "${root}/build.rs" -o "${work}/ban_scanner"; then
  echo "ban_scanner: build.rs did not compile" >&2
  exit 2
fi

# stdout is the `cargo:` directive stream (thousands of rerun-if-changed lines);
# stderr is the human report. Only the report is worth showing.
OUT_DIR="${work}/out" CARGO_MANIFEST_DIR="${root}" \
  "${work}/ban_scanner" >"${work}/directives" 2>"${work}/report"
status=$?

cat "${work}/report" >&2

python3 "${root}/scripts/spec_ban_ratchet.py" --root "${root}"
ratchet_status=$?

# 2 (could not measure) outranks 1 (violations): a scanner that did not run
# has certified nothing, whatever the other one found.
if [ "${status}" -eq 2 ] || [ "${ratchet_status}" -eq 2 ]; then
  exit 2
fi
if [ "${status}" -ne 0 ] || [ "${ratchet_status}" -ne 0 ]; then
  exit 1
fi
exit 0
