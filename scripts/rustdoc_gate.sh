#!/usr/bin/env bash
#
# Does every crate in the workspace document with ZERO rustdoc errors? (#2753)
#
# WHY THIS EXISTS, AND WHY ITS BAR IS ZERO
# ----------------------------------------
# `cargo doc` exits 101 on this workspace and no CI job had ever noticed
# (#2711): nothing in `.github/` ran `cargo doc` or `rustdoc` at all, and
# `docs.yml` -- mkdocs, green every night -- said nothing about it. The errors
# exist because `[lints.rust] warnings = "deny"` is forwarded to rustdoc as
# well as rustc, so rustdoc's default-`warn` lints
# (`rustdoc::private_intra_doc_links`, `rustdoc::broken_intra_doc_links`)
# become hard errors that `cargo build` and `cargo test` never see.
#
# #2753 fixed every rustdoc error in the workspace, so the bar is ZERO for
# every crate and no list of tolerated offenders exists.
#
# THE COVERAGE CONTROL, AND WHY A GREEN HERE CANNOT BE A NON-RUN
# ---------------------------------------------------------------
# The ledger this gate replaced was its own positive control: known-failing
# crates run through the same command every time, so a scan that silently did
# nothing would report them "clean" and fail loudly. A zero bar has no such
# inputs. With zero errors everywhere, `rc=0, errors=0` for every crate is
# byte-identical to a scan that never ran rustdoc at all -- wrong flags, no
# toolchain, an empty crate list, a typo in a package name.
#
# So the control is rebuilt from the tool's own output instead. For every
# crate this script requires the rustdoc OUTPUT DIRECTORY to exist after the
# pass -- `<target-dir>/doc/<target-name>/index.html`, where both the target
# directory and the per-package target names come from `cargo metadata`, never
# from a list written down here. A crate whose docs are absent was not
# documented, and that is reported as a NON-RUN for that crate with its own
# message, distinct from a rustdoc error. Adding a crate to the workspace
# extends the control automatically; renaming one cannot leave a stale
# expectation behind.
#
# The doc tree is deleted before the scan so the check cannot be satisfied by
# an earlier run's output.
#
# USAGE
#   scripts/rustdoc_gate.sh
# Exit 0 = every crate documented, with zero rustdoc errors.
#      1 = a crate has rustdoc errors, or a crate's docs were never produced.
#      2 = could not measure (treat as a NON-RUN, never as a pass).

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Same exclusion as cross-check.yml's compile jobs and the doctest gate:
# building gam-pyffi needs a configured Python interpreter for pyo3-ffi.
# Excluded is NOT a claim of cleanliness, and this script says so out loud.
EXCLUDED_CRATES=("gam-pyffi")

die() {
  echo "rustdoc-gate: FATAL: $*" >&2
  exit 2
}

command -v cargo >/dev/null 2>&1 || die "cargo not on PATH"
command -v jq >/dev/null 2>&1 || die "jq not on PATH (needed to read cargo metadata)"

cd "${REPO_ROOT}" || die "cannot cd to ${REPO_ROOT}"

META="$(mktemp)" || die "cannot create temp file"
trap 'rm -f "${META}"' EXIT
cargo metadata --no-deps --format-version 1 >"${META}" 2>/dev/null || die "cargo metadata failed"

TARGET_DIR="$(jq -r '.target_directory' "${META}")"
[ -n "${TARGET_DIR}" ] && [ "${TARGET_DIR}" != "null" ] || die "cargo metadata has no target_directory"

# `while read` rather than `mapfile`: mapfile is bash 4 and the dev host is
# bash 3.2. A gate that only runs where it is deployed cannot be given a
# positive/negative control before it lands -- that is exactly how #2732's
# gate was demonstrated to fire.
CRATES=()
while IFS= read -r line; do
  [ -n "$line" ] && CRATES+=("$line")
done < <(jq -r '.packages[].name' "${META}" | sort)
[ "${#CRATES[@]}" -gt 0 ] || die "cargo metadata returned no workspace packages"

is_excluded() {
  local needle="$1" c
  for c in "${EXCLUDED_CRATES[@]}"; do
    [ "$c" = "$needle" ] && return 0
  done
  return 1
}

# Every documentable target of a package, as the DIRECTORY NAME rustdoc writes.
# rustdoc replaces `-` with `_`; bin, lib, rlib and proc-macro targets all get
# a directory, and tests/benches/examples do not.
doc_dirs_for() {
  jq -r --arg p "$1" '
    .packages[] | select(.name == $p) | .targets[]
    | select([.kind[]] | any(. == "lib" or . == "rlib" or . == "proc-macro" or . == "bin"))
    | .name | gsub("-"; "_")
  ' "${META}" | sort -u
}

echo "rustdoc-gate: target directory ${TARGET_DIR}"
echo "rustdoc-gate: excluded from this scan: ${EXCLUDED_CRATES[*]} (not measured, not a cleanliness claim)"

# Delete the doc tree first. Without this, the coverage control below could be
# satisfied by an index.html some EARLIER run wrote, which is precisely the
# "a green that is really a non-run" failure it exists to catch.
rm -rf "${TARGET_DIR}/doc" || die "cannot remove ${TARGET_DIR}/doc"

SCANNED=0
RED=()
UNDOCUMENTED=()

echo
for crate in "${CRATES[@]}"; do
  if is_excluded "$crate"; then
    printf '  %-24s EXCLUDED (not measured, not a cleanliness claim)\n' "$crate"
    continue
  fi
  log="$(mktemp)"
  cargo doc -p "$crate" --no-deps >"$log" 2>&1
  rc=$?
  # Strip ANSI before counting. Cargo colours diagnostics whenever it decides
  # the sink is a terminal, and `^error` then fails to match a line that really
  # begins with an escape sequence -- which is how the first per-crate sweep of
  # this surface reported `errors=0` beside `rc=101` on all fifteen red crates.
  sed -e 's/\x1b\[[0-9;]*[a-zA-Z]//g' "$log" >"${log}.plain"
  errors="$(grep -cE '^error(\[|:)' "${log}.plain")"
  SCANNED=$((SCANNED + 1))

  # --- coverage control, per crate ---------------------------------------
  missing_dirs=""
  ndirs=0
  while IFS= read -r d; do
    [ -n "$d" ] || continue
    ndirs=$((ndirs + 1))
    [ -s "${TARGET_DIR}/doc/${d}/index.html" ] || missing_dirs="${missing_dirs} ${d}"
  done < <(doc_dirs_for "$crate")

  if [ "$rc" -ne 0 ] || [ "$errors" -gt 0 ]; then
    RED+=("$crate")
    printf '  %-24s RED       rc=%s errors=%s\n' "$crate" "$rc" "$errors"
    if [ "$errors" -eq 0 ]; then
      echo "      (count unreliable: rc=${rc} with no matched error lines; tail follows)"
      tail -n 8 "${log}.plain" | sed 's/^/      /'
    else
      grep -E '^error(\[|:)' "${log}.plain" | head -n 3 | sed 's/^/      /'
    fi
  elif [ "$ndirs" -eq 0 ] || [ -n "$missing_dirs" ]; then
    UNDOCUMENTED+=("${crate}:${missing_dirs:-<no documentable target>}")
    printf '  %-24s NON-RUN   rc=0 but no docs at %s/doc/%s\n' \
      "$crate" "${TARGET_DIR}" "${missing_dirs:-<no documentable target>}"
  else
    printf '  %-24s ok        rc=0 errors=0 docs=%s dir(s)\n' "$crate" "$ndirs"
  fi
  rm -f "$log" "${log}.plain"
done

echo
echo "rustdoc-gate: scanned ${SCANNED} crates: $((SCANNED - ${#RED[@]} - ${#UNDOCUMENTED[@]})) documented clean, ${#RED[@]} with errors, ${#UNDOCUMENTED[@]} with no docs produced"

status=0

if [ "${#RED[@]}" -gt 0 ]; then
  status=1
  echo
  echo "::error title=rustdoc errors::These crates do not document cleanly: ${RED[*]}"
  for c in "${RED[@]}"; do
    echo "  Reproduce with: cargo doc -p ${c} --no-deps"
  done
  echo "  The bar is ZERO, deliberately: this surface was dark for the whole project's history (#2711),"
  echo "  and an allowance at N licenses the N+1'th. Fix the link; there is no allowance file."
fi

if [ "${#UNDOCUMENTED[@]}" -gt 0 ]; then
  status=1
  echo
  echo "::error title=rustdoc coverage gap::cargo doc exited 0 but produced no documentation for: ${UNDOCUMENTED[*]}"
  echo "  This run therefore says NOTHING about those crates. It is a non-run for them, not a pass."
fi

if [ "$status" -eq 0 ]; then
  echo "rustdoc-gate: OK -- every crate documented (index.html produced for each target) with zero rustdoc errors."
fi
exit "$status"
