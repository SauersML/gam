#!/usr/bin/env bash
#
# precheck.sh — fast pre-push gate for the gam build.rs ban rules.
#
# WHY: build.rs fails the build on a family of banned patterns (let _,
# debug_assert!, #[allow]/#[expect] of non-clippy lints, println! in src,
# underscore fn params, #[cfg(test)] on src items, bare #[should_panic],
# #[ignore], cfg(feature) gates, assertion-less #[test], and the wider
# lexical-substring ban list). A full `cargo build`
# surfaces these only after a ~15-minute compile. This script reproduces the
# *text-scan* tier of those rules in seconds with NO compiler, so a violation is
# caught at the keyboard instead of a wheel-cycle later.
#
# It is a FIRST-LINE filter, not a full replacement for build.rs: the deep
# cross-file rules (src item referenced only by tests, pub(crate) item with zero
# consumers, dodge-named / no-op-sentinel fns) require whole-crate analysis and
# are left to build.rs / CI. Everything this script flags IS a real build.rs
# failure; a clean run here does not prove build.rs is green, but it catches the
# patterns that actually cause the recurring wheel-cycle failures.
#
# USAGE:
#   scripts/precheck.sh              # fast text scan only (default; seconds)
#   scripts/precheck.sh --check      # text scan, then incremental `cargo check`
#   scripts/precheck.sh --check-only # skip text scan, just `cargo check`
#
# Exit code 0 = clean, 1 = at least one violation (or cargo check failure).

set -euo pipefail

# Resolve repo root from this script's location so it works from any cwd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODE="scan"
case "${1:-}" in
    --check) MODE="scan+check" ;;
    --check-only) MODE="check" ;;
    "" ) MODE="scan" ;;
    -h|--help)
        sed -n '2,33p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
        exit 0
        ;;
    *)
        echo "precheck.sh: unknown argument '$1' (use --check, --check-only, or no args)" >&2
        exit 2
        ;;
esac

run_text_scan() {
    PYBIN="$(command -v python3 || command -v python || true)"
    if [ -z "${PYBIN}" ]; then
        echo "precheck.sh: python3 not found; cannot run the text scan" >&2
        return 2
    fi
    "${PYBIN}" "${SCRIPT_DIR}/precheck_scan.py" "${REPO_ROOT}"
}

run_cargo_check() {
    if ! command -v cargo >/dev/null 2>&1; then
        echo "precheck.sh: cargo not on PATH; skipping --check tier" >&2
        return 0
    fi
    echo "precheck.sh: incremental cargo check (niced; this can take a while on a cold cache)..." >&2
    # Keep it light on a shared machine: nice + single check, short messages.
    nice -n 19 cargo check --workspace --message-format short 2>&1 || return 1
}

rc=0
case "${MODE}" in
    scan)        run_text_scan || rc=$? ;;
    "scan+check")
        run_text_scan || rc=$?
        if [ "${rc}" -eq 0 ]; then run_cargo_check || rc=$?; fi
        ;;
    check)       run_cargo_check || rc=$? ;;
esac
exit "${rc}"
