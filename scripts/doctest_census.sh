#!/usr/bin/env bash
#
# Which crate's doctests fail? (#2732) A MEASUREMENT, NOT A VERDICT.
#
# WHAT THIS IS, AND WHAT GATES INSTEAD
# ------------------------------------
# The GATE is `scripts/doctest_gate.sh`: one `cargo test --doc --workspace`
# pass, wired into `cross-check.yml` on push, at a bar of ZERO -- any failing
# doctest fails it, and there is no ledger of tolerated crates. This file
# answers the question the gate cannot: WHICH crate went red. It shells
# `cargo test --doc -p <crate>` once per crate, which measured ~1h39m over 24
# crates (run 30696464338), so it is a triage cost rather than a push cost.
#
# It gates nothing and exits 0 once the sweep completes, whatever it found.
# Reading its table is the point. It is refused under `push` so it can never
# become a job that always passes.
#
# WHY IT IS SEPARATE FROM `scripts/rustdoc_gate.sh`
# -------------------------------------------------
# That one runs `cargo doc -p <crate> --no-deps` -- doc GENERATION. This runs
# `cargo test --doc -p <crate>` -- doc TESTS. Different command, different
# surface, and neither result predicts the other: measured at c3635e04c,
# `cargo test --doc -p gam-solve` was `0 passed; 12 failed` while
# `git grep -rn "test --doc" -- .github scripts` returned nothing at all. One
# green covering both commands would make both uninterpretable, even though one
# root cause (Unicode in indented `///` blocks that rustdoc compiles as Rust)
# feeds both: fencing a block as ```text fixes the doctest without touching
# whatever the doc build objects to.
#
# Exit 0 = the sweep completed. 2 = it could not measure (treat as a NON-RUN).

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Same exclusion as cross-check.yml's compile jobs and the doctest gate:
# building gam-pyffi needs a configured Python interpreter for pyo3-ffi.
# Excluded is NOT a claim of cleanliness, and the sweep says so per crate.
EXCLUDED_CRATES=("gam-pyffi")

die() {
  echo "doctest-census: FATAL: $*" >&2
  exit 2
}

command -v cargo >/dev/null 2>&1 || die "cargo not on PATH"
command -v jq >/dev/null 2>&1 || die "jq not on PATH (needed to read cargo metadata)"

if [ "${GITHUB_EVENT_NAME:-}" = "push" ]; then
  die "this is a measurement, not a verdict, and must not run as a push gate"
fi

cd "${REPO_ROOT}" || die "cannot cd to ${REPO_ROOT}"

# Only crates with a LIB target can have doctests. Asking `cargo test --doc -p
# <crate>` of a bin-only crate is a category error, not a finding: the first
# census (run 30696464338) reported gam-cli as FAILING at rc=101 when the whole
# message was `no library targets found in package gam-cli`. gam-cli is a
# binary. It has no doctests and never had any; filtering on the lib target
# removes the artifact at its source rather than special-casing the name.
# A `while read` loop rather than `mapfile`, so this runs on a bash 3.2 host
# (macOS) as well as on the runner.
ALL_CRATES=()
while IFS= read -r line; do
  [ -n "$line" ] && ALL_CRATES+=("$line")
done < <(
  cargo metadata --no-deps --format-version 1 \
    | jq -r '.packages[] | select([.targets[].kind[]] | any(. == "lib" or . == "rlib" or . == "proc-macro")) | .name' \
    | sort
)
[ "${#ALL_CRATES[@]}" -gt 0 ] || die "cargo metadata returned no workspace lib packages"

is_excluded() {
  local needle="$1" c
  for c in "${EXCLUDED_CRATES[@]}"; do
    [ "$c" = "$needle" ] && return 0
  done
  return 1
}

echo "=============================================================="
echo " doctest CENSUS -- THIS IS A MEASUREMENT, NOT A VERDICT"
echo " Nothing below gates anything. The gate is scripts/doctest_gate.sh,"
echo " one --workspace pass at a bar of zero."
echo "=============================================================="
echo "doctest-census: excluded from this scan: ${EXCLUDED_CRATES[*]}"
echo

SCANNED=0
FAILING=()
PASSING=()

for crate in "${ALL_CRATES[@]}"; do
  if is_excluded "$crate"; then
    printf '  %-24s EXCLUDED (not measured, not a cleanliness claim)\n' "$crate"
    continue
  fi
  log="$(mktemp)"
  cargo test --doc -p "$crate" >"$log" 2>&1
  rc=$?
  # Strip ANSI before matching: cargo colours its output when it decides the
  # sink is a terminal, and a `^test result:` match then fails against a line
  # that begins with an escape sequence.
  sed -e 's/\x1b\[[0-9;]*[a-zA-Z]//g' "$log" >"${log}.plain"
  result_line="$(grep -E '^test result:' "${log}.plain" | tail -n 1)"
  SCANNED=$((SCANNED + 1))
  if [ "$rc" -eq 0 ]; then
    PASSING+=("$crate")
    printf '  %-24s pass   rc=0   %s\n' "$crate" "${result_line:-(no doctests)}"
  else
    FAILING+=("$crate")
    printf '  %-24s FAIL   rc=%s   %s\n' "$crate" "$rc" "${result_line:-(no test result line)}"
    if [ -z "${result_line}" ]; then
      # A non-zero exit with no `test result:` line means the doctest harness
      # never reported -- a build failure, not a failing doctest. Those are
      # different findings with different owners, so do not let one read as the
      # other.
      echo "      (no test result line: the harness did not run. Build failure tail:)"
      grep -E '^error' "${log}.plain" | tail -n 3 | sed 's/^/      /'
    else
      echo "      Reproduce: cargo test --doc -p ${crate}"
      echo "      Note an INDENTED block in a /// comment is compiled as Rust. If it is prose or math, fence it as a text block."
    fi
  fi
  rm -f "$log" "${log}.plain"
done

echo
echo "doctest-census: scanned ${SCANNED} crates: ${#PASSING[@]} passing, ${#FAILING[@]} failing"
if [ "${#FAILING[@]}" -gt 0 ]; then
  echo "doctest-census: failing crates: ${FAILING[*]}"
fi
echo "CENSUS COMPLETE -- gated nothing."
exit 0
