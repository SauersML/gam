#!/usr/bin/env bash
#
# Does every doctest in the workspace still compile and pass? (#2732)
#
# WHY THIS EXISTS
# ---------------
# `cargo test --doc` ran in NO CI job. `git grep -rn "test --doc" -- .github
# scripts` returned nothing, so the entire doctest surface was dark by
# omission -- and it was red: the workspace census (run 30696464338, 24 crates,
# 1h39m) measured 25 failing doctests in four crates:
#
#     gam-solve   12      gam-terms    8
#     gam-models   3      gam-sae      2
#
# All 25 were the same defect. An INDENTED block inside a `///` or `//!`
# comment is a Markdown indented code block, and rustdoc compiles those as
# Rust. Twenty-five blocks of ASCII/Unicode mathematics were therefore being
# handed to rustc, which died on `½`, `·`, `⁺`, `ᵀ`, `δ`, `ρ`. They are fenced
# as ```text now, and the workspace is green. The bar here is ZERO: any failing
# doctest fails this gate, and there is no ledger of tolerated crates.
# This is the gate.
#
# WHY IT IS ONE WORKSPACE PASS AND NOT 24 PER-CRATE PASSES
# A per-crate `cargo test --doc -p <crate>` sweep took ~1h39m, which is a
# triage cost, not a push-gate cost. One `--workspace`
# invocation builds the dependency graph once.
#
# THE COVERAGE CONTROL, AND WHY A GREEN HERE CANNOT BE A NON-RUN
# ---------------------------------------------------------------
# The workspace currently contains ZERO doctests. Every crate reports
# `0 passed; 0 failed`. That is a real verdict about a real surface -- it says
# no doc comment in the tree accidentally compiles as broken Rust -- but it is
# also, byte for byte, what a scan that silently measured NOTHING would print.
#
# So the pass/fail of the doctests is not the only thing checked. This script
# enumerates every workspace crate that HAS a lib target (from `cargo
# metadata`, so the list cannot go stale) and requires a `Doc-tests <crate>`
# banner for each one. A missing banner means that crate's doctests were never
# collected, and that is a failure with its own message, distinct from a
# doctest failing. The expected set is derived, never written down: adding a
# crate extends the control automatically.
#
# Deriving it from lib targets also fixes a real misreading in the census:
# `cargo test --doc -p gam-cli` exits 101 with `no library targets found in
# package gam-cli`, and the census reported that as a FAILING crate. gam-cli is
# a binary-only crate. It has no doctests to run and never did; the red was an
# artifact of asking a bin-only crate a lib-only question.
#
# Exit 0 = every crate's doctests were collected and all passed.
#      1 = a doctest failed, or a crate's doctests were never collected.
#      2 = could not measure (treat as a NON-RUN, never as a pass).

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Same exclusion as cross-check.yml's compile jobs, the rustdoc gate and the
# doctest census: building gam-pyffi needs a configured Python interpreter for
# pyo3-ffi. Excluded is NOT a claim of cleanliness, and this script says so.
EXCLUDED_CRATES=("gam-pyffi")

die() {
  echo "doctest-gate: FATAL: $*" >&2
  exit 2
}

command -v cargo >/dev/null 2>&1 || die "cargo not on PATH"
command -v jq >/dev/null 2>&1 || die "jq not on PATH (needed to read cargo metadata)"

cd "${REPO_ROOT}" || die "cannot cd to ${REPO_ROOT}"

# Only crates with a LIB target can have doctests. `cargo test --doc` is a
# lib-only question and asking it of a bin-only crate is a category error, not
# a finding.
# A `while read` loop rather than `mapfile`, so this script can be exercised on
# a bash 3.2 host (macOS) as well as on the runner. A gate that only runs where
# it is deployed cannot be given a positive/negative control before it lands.
LIB_CRATES=()
while IFS= read -r line; do
  [ -n "$line" ] && LIB_CRATES+=("$line")
done < <(
  cargo metadata --no-deps --format-version 1 \
    | jq -r '.packages[] | select([.targets[].kind[]] | any(. == "lib" or . == "rlib" or . == "proc-macro")) | .name' \
    | sort
)
[ "${#LIB_CRATES[@]}" -gt 0 ] || die "cargo metadata returned no workspace lib crates"

EXCLUDE_ARGS=()
EXPECTED=()
for crate in "${LIB_CRATES[@]}"; do
  skip=0
  for x in "${EXCLUDED_CRATES[@]}"; do
    [ "$x" = "$crate" ] && skip=1
  done
  if [ "$skip" -eq 1 ]; then
    EXCLUDE_ARGS+=(--exclude "$crate")
  else
    EXPECTED+=("$crate")
  fi
done

echo "doctest-gate: excluded from this scan: ${EXCLUDED_CRATES[*]} (not measured, not a cleanliness claim)"
echo "doctest-gate: expecting doctest collection for ${#EXPECTED[@]} lib crate(s)"
echo

LOG="$(mktemp)" || die "cannot create temp file"
trap 'rm -f "${LOG}" "${LOG}.plain"' EXIT

# One invocation, streamed to the job log for a human AND captured for the
# control. PIPESTATUS carries cargo's exit code past the tee.
cargo test --doc --locked --workspace "${EXCLUDE_ARGS[@]}" 2>&1 | tee "${LOG}"
rc="${PIPESTATUS[0]}"

# Strip ANSI before matching. cargo colours its output when it decides the sink
# is a terminal, and `Doc-tests` then fails to match a line that begins with an
# escape sequence -- the first per-crate rustdoc sweep reported `errors=0`
# beside `rc=101` on all fifteen red crates for exactly that reason.
sed -e 's/\x1b\[[0-9;]*[a-zA-Z]//g' "${LOG}" >"${LOG}.plain"

status=0
if [ "$rc" -ne 0 ]; then
  status=1
  echo
  echo "::error title=Doctest failure::cargo test --doc exited ${rc}. A doctest in this workspace does not compile or does not pass."
  echo "  Reproduce: cargo test --doc --locked --workspace ${EXCLUDE_ARGS[*]}"
  echo "  Note an INDENTED block in a /// or //! comment is compiled as Rust."
  echo "  If it is prose or mathematics, fence it as a text block; do not mark it ignore."
fi

# --- coverage control -------------------------------------------------------
# Every expected crate must have had its doctests COLLECTED. Without this, a
# scan that built nothing prints the same green as a scan that checked
# everything, because the workspace's honest answer is `0 passed; 0 failed`.
MISSING=()
for crate in "${EXPECTED[@]}"; do
  # cargo prints `   Doc-tests <name>` per lib crate. Accept both spellings of
  # the name: the banner carries the lib TARGET name, and a hyphenated package
  # may present either as `gam-solve` or as the `gam_solve` rustc crate name
  # depending on how the target was declared. Matching one spelling only would
  # turn a naming detail into a fake coverage gap.
  hyphen="${crate//_/-}"
  under="${crate//-/_}"
  if ! grep -qE "^[[:space:]]*Doc-tests[[:space:]]+(${hyphen}|${under})[[:space:]]*\$" "${LOG}.plain"; then
    MISSING+=("$crate")
  fi
done

echo
echo "doctest-gate: collected doctests for $(( ${#EXPECTED[@]} - ${#MISSING[@]} )) of ${#EXPECTED[@]} expected lib crate(s)"

if [ "${#MISSING[@]}" -gt 0 ]; then
  status=1
  echo
  echo "::error title=Doctest coverage gap::No 'Doc-tests' banner for: ${MISSING[*]}"
  echo "  Those crates' doctests were never collected, so this run says nothing about them."
  echo "  That is a NON-RUN for those crates, not a pass. Do not read the green above as covering them."
fi

if [ "$status" -eq 0 ]; then
  echo "doctest-gate: OK -- every expected crate's doctests were collected and every doctest passed."
fi
exit "$status"
